# direct_ble_to_lsl_gain_params.py
import asyncio
import time
import logging
import argparse
from bleak import BleakScanner, BleakClient
from pylsl import StreamInfo, StreamOutlet, local_clock

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SERVICE_UUID   = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
DATA_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
CMD_CHAR_UUID  = "c0de0001-36e1-4688-b7f5-ea07361b26a8"

CHANNELS_PER_NODE = 16
PACKET_SIZE = 51

SAMPLING_RATE = 250.0  # Честная частота АЦП (250 Гц)
SAMPLE_DT = 1.0 / SAMPLING_RATE  # 0.004 сек между сэмплами

DEFAULT_GAIN = 8

GAIN_REGISTER_MAP = {
    1:   0x0000,
    2:   0x1111,
    4:   0x2222,
    8:   0x3333,
    16:  0x4444,
    32:  0x5555,
    64:  0x6666,
    128: 0x7777
}

CONNECTION_LOCK = asyncio.Lock()

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# ==============================================================================
# DEVICE STATE MANAGER (C REALTME LSL TIMESTAMPS)
# ==============================================================================
class DeviceState:
    """ Tracks connection statistics, LSL outlets, and Auto-Tuner parameters. """
    def __init__(self, mac_address):
        self.mac_address = mac_address
        self.clean_mac = mac_address.replace(":", "").replace("-", "")
        self.is_connected = False
        
        # Statistics
        self.last_counter = -1
        self.packets_received = 0
        self.total_received = 0
        self.total_lost = 0
        self.lost_this_second = 0
        
        # Auto-Tuner state
        self.global_chop_enabled = False
        self.last_tune_time = time.time()
        
        # CMD Response Queue
        self.cmd_response_event = asyncio.Event()
        self.last_cmd_response = None
        
        # Real-time LSL Clock Tracker (Восстановление идеального времени)
        self.last_lsl_timestamp = 0.0
        
        # LSL Initialization c ЧЕСТНОЙ ЧАСТОТОЙ 250 Гц
        logger.info(f"Creating Real-Time LSL Stream 'FreeEEG_{self.clean_mac}' at {SAMPLING_RATE} Hz...")
        info = StreamInfo(
            name=f'FreeEEG_{self.clean_mac}', 
            type='EEG', 
            channel_count=CHANNELS_PER_NODE, 
            nominal_srate=SAMPLING_RATE, # ЧЕСТНЫЕ 250 Гц ДЛЯ СГЛАЖИВАНИЯ ДЖИТТЕРА
            channel_format='int32', 
            source_id=f'uid_{self.mac_address}'
        )
        self.outlet = StreamOutlet(info)

active_devices = {}

# ==============================================================================
# REAL-TIME NOTIFICATION HANDLER
# ==============================================================================
def create_notification_handler(mac_str):
    """ Handler for high-speed EEG data stream with Sub-Millisecond LSL Timestamps. """
    state = active_devices[mac_str]
    
    def notification_handler(sender: int, data: bytearray):
        if len(data) == PACKET_SIZE and data[0] == 0xA0 and data[50] == 0xC0:
            counter = data[1]
            
            # Быстрый перевод 24-bit в signed 32-bit int
            channels = [
                int.from_bytes(data[2 + i*3 : 5 + i*3], byteorder='big', signed=True)
                for i in range(CHANNELS_PER_NODE)
            ]
            
            # --- РЕАЛТАЙМ-МАТЕМАТИКА ВРЕМЕНИ LSL ---
            now_lsl = local_clock()
            
            # Если это первый сэмпл или был сбой > 50 мс, привязываемся к текущим часам
            if state.last_lsl_timestamp == 0.0 or abs(now_lsl - state.last_lsl_timestamp) > 0.050:
                sample_time = now_lsl
            else:
                # Плавно двигаем время вперед ровно на 4 мс (1/250 Гц)
                sample_time = state.last_lsl_timestamp + SAMPLE_DT
                
            state.last_lsl_timestamp = sample_time
            
            # Отправляем сэмпл в LSL с точнейшим квантованным таймстемпом
            state.outlet.push_sample(channels, timestamp=sample_time)
            
            # Подсчет потерь
            if state.last_counter != -1:
                expected_counter = (state.last_counter + 1) % 256
                if counter != expected_counter:
                    loss = (counter - expected_counter) % 256
                    state.total_lost += loss
                    state.lost_this_second += loss
            
            state.last_counter = counter
            state.packets_received += 1
            state.total_received += 1
            
    return notification_handler

def create_cmd_response_handler(mac_str):
    state = active_devices[mac_str]
    def cmd_handler(sender: int, data: bytearray):
        if len(data) >= 3:
            reg_addr = data[0]
            reg_val = (data[1] << 8) | data[2]
            state.last_cmd_response = (reg_addr, reg_val)
            state.cmd_response_event.set()
    return cmd_handler

# ==============================================================================
# GAIN CONFIGURATION & VERIFICATION
# ==============================================================================
async def set_and_verify_gain(client: BleakClient, mac_str: str, gain: int, max_retries: int = 3) -> bool:
    if gain not in GAIN_REGISTER_MAP:
        logger.error(f"Invalid gain value: {gain}. Supported: {list(GAIN_REGISTER_MAP.keys())}")
        return False
        
    reg_val = GAIN_REGISTER_MAP[gain]
    state = active_devices[mac_str]
    
    logger.info(f"[{mac_str}] Configuring PGA Gain = {gain} (Register Value: 0x{reg_val:04X})...")
    
    for attempt in range(1, max_retries + 1):
        try:
            write_gain1_payload = bytearray([0x04, (reg_val >> 8) & 0xFF, reg_val & 0xFF])
            await client.write_gatt_char(CMD_CHAR_UUID, write_gain1_payload, response=False)
            await asyncio.sleep(0.08)
            
            state.cmd_response_event.clear()
            await client.write_gatt_char(CMD_CHAR_UUID, bytearray([0x04]), response=False)
            
            await asyncio.wait_for(state.cmd_response_event.wait(), timeout=1.5)
            r_addr1, r_val1 = state.last_cmd_response
            
            write_gain2_payload = bytearray([0x05, (reg_val >> 8) & 0xFF, reg_val & 0xFF])
            await client.write_gatt_char(CMD_CHAR_UUID, write_gain2_payload, response=False)
            await asyncio.sleep(0.08)
            
            state.cmd_response_event.clear()
            await client.write_gatt_char(CMD_CHAR_UUID, bytearray([0x05]), response=False)
            
            await asyncio.wait_for(state.cmd_response_event.wait(), timeout=1.5)
            r_addr2, r_val2 = state.last_cmd_response
            
            if r_addr1 == 0x04 and r_val1 == reg_val and r_addr2 == 0x05 and r_val2 == reg_val:
                logger.info(f"[{mac_str}] ✓ PGA Gain = {gain} VERIFIED! (REG 0x04=0x{r_val1:04X}, REG 0x05=0x{r_val2:04X})")
                return True
        except Exception as e:
            logger.error(f"[{mac_str}] Error setting gain: {e}")
        await asyncio.sleep(0.2)
        
    return False

async def send_global_chop_command(client: BleakClient, enable: bool):
    payload = bytearray([0x06, 0x01 if enable else 0x00, 0x00, 0x01, 0x00])
    try:
        await client.write_gatt_char(CMD_CHAR_UUID, payload, response=False)
    except Exception as e:
        logger.error(f"Failed to send Global-Chop command: {e}")

async def handle_device_connection(device, target_gain: int):
    mac = device.address
    if mac not in active_devices:
        active_devices[mac] = DeviceState(mac)
    state = active_devices[mac]
    
    async with CONNECTION_LOCK:
        logger.info(f"Attempting connection to {mac}...")
        try:
            client = BleakClient(device, timeout=15.0)
            await client.connect()
        except Exception as e:
            logger.error(f"Connection failed for {mac}: {e}")
            return
            
    try:
        state.is_connected = True
        logger.info(f"<<< CONNECTED: {mac} >>>")
        
        await client.start_notify(CMD_CHAR_UUID, create_cmd_response_handler(mac))
        await set_and_verify_gain(client, mac, gain=target_gain)
        await send_global_chop_command(client, enable=False)
        state.global_chop_enabled = False
        
        await client.start_notify(DATA_CHAR_UUID, create_notification_handler(mac))
        logger.info(f"[{mac}] Real-Time LSL Streaming Active.")
        
        while client.is_connected:
            await asyncio.sleep(1.0)
            state.lost_this_second = 0 
            
    except Exception as e:
        logger.error(f"Connection dropped for {mac}: {e}")
    finally:
        state.is_connected = False
        state.last_counter = -1

async def stats_logger():
    while True:
        await asyncio.sleep(1.0)
        connected_nodes = [s for s in active_devices.values() if s.is_connected]
        if connected_nodes:
            print("\n--- Direct BLE-to-LSL Real-Time Stats ---")
            for state in connected_nodes:
                print(f"Device [{state.mac_address}] | Rx: {state.packets_received} pkts/s (250Hz nominal) | Lost: {state.total_lost}")
                state.packets_received = 0

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gain', type=int, default=DEFAULT_GAIN, choices=[1, 2, 4, 8, 16, 32, 64, 128])
    args = parser.parse_args()

    logger.info(f"Initializing Real-Time BLE-to-LSL Bridge (PGA Gain = {args.gain})...")
    asyncio.create_task(stats_logger())
    managed_macs = set()

    def detection_callback(device, advertisement_data):
        mac = device.address
        uuids = advertisement_data.service_uuids
        if uuids and any(SERVICE_UUID.lower() in str(u).lower() for u in uuids):
            if mac not in managed_macs:
                if mac not in active_devices or not active_devices[mac].is_connected:
                    managed_macs.add(mac)
                    async def task_wrapper():
                        await handle_device_connection(device, target_gain=args.gain)
                        managed_macs.remove(mac)
                    asyncio.create_task(task_wrapper())

    async with BleakScanner(detection_callback=detection_callback):
        while True:
            await asyncio.sleep(3600.0)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
