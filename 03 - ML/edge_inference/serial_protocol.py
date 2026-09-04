"""
Framed Binary UART Protocol & CRC16 Decoder (ADR-001 & PLAN §0)
==============================================================
Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
        via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)

Packet Structure (19 bytes):
  Byte 0:  0xAA (FRAME_START_BYTE)
  Byte 1:  0x01 (FRAME_TYPE_RAW)
  Bytes 2-5:   timestamp_ms (uint32_t, little-endian)
  Bytes 6-9:   ir_raw       (uint32_t, little-endian)
  Bytes 10-13: red_raw      (uint32_t, little-endian)
  Bytes 14-15: heuristic_bpm_x10 (uint16_t, little-endian)
  Bytes 16-17: crc16        (uint16_t, little-endian, CRC-16-CCITT over bytes 1..15)
  Byte 18: 0x55 (FRAME_END_BYTE)
"""

import struct
from typing import Optional, Tuple, Dict, Any

FRAME_START_BYTE = 0xAA
FRAME_END_BYTE = 0x55
FRAME_TYPE_RAW = 0x01
PACKET_SIZE = 19
PAYLOAD_SIZE = 15  # Bytes 1 to 15 inclusive


def compute_crc16(data: bytes) -> int:
    """
    CRC-16-CCITT (Polynomial 0x1021, Initial value 0xFFFF).
    Matches the firmware computeCRC16 implementation in main.cpp.
    """
    crc = 0xFFFF
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def pack_ppg_frame(timestamp_ms: int, ir_raw: int, red_raw: int, heuristic_bpm: float = 0.0) -> bytes:
    """Pack raw sample into 19-byte binary frame."""
    bpm_x10 = int(round(heuristic_bpm * 10.0)) & 0xFFFF
    payload = struct.pack("<BIIIH", FRAME_TYPE_RAW, timestamp_ms, ir_raw, red_raw, bpm_x10)
    crc = compute_crc16(payload)
    packet = struct.pack("<B", FRAME_START_BYTE) + payload + struct.pack("<HB", crc, FRAME_END_BYTE)
    return packet


class StreamPacketParser:
    """
    Streaming byte-by-byte parser with resynchronization and CRC16 validation.
    Survives framing errors, dropped bytes, and UART noise.
    """
    def __init__(self):
        self.buffer = bytearray()

    def feed(self, raw_bytes: bytes):
        self.buffer.extend(raw_bytes)

    def parse_next(self) -> Optional[Dict[str, Any]]:
        """
        Attempts to extract the next valid frame from the buffer.
        Returns None if not enough bytes or awaiting alignment.
        """
        while len(self.buffer) >= PACKET_SIZE:
            # 1. Search for start delimiter
            start_idx = self.buffer.find(bytes([FRAME_START_BYTE]))
            if start_idx == -1:
                # No start delimiter in buffer, discard all
                self.buffer.clear()
                return None

            if start_idx > 0:
                # Discard noise bytes preceding start delimiter
                del self.buffer[:start_idx]
                if len(self.buffer) < PACKET_SIZE:
                    return None

            # 2. Check end delimiter
            if self.buffer[PACKET_SIZE - 1] != FRAME_END_BYTE:
                # False start byte, skip this byte and search again
                del self.buffer[0]
                continue

            # 3. Extract candidate frame
            candidate = bytes(self.buffer[:PACKET_SIZE])
            payload = candidate[1:1 + PAYLOAD_SIZE]
            received_crc = struct.unpack("<H", candidate[16:18])[0]

            # 4. Verify CRC16
            expected_crc = compute_crc16(payload)
            if expected_crc != received_crc:
                # CRC error, discard start byte and resync
                del self.buffer[0]
                continue

            # 5. Valid packet! Unpack payload
            frame_type, ts_ms, ir_raw, red_raw, bpm_x10 = struct.unpack("<BIIIH", payload)
            del self.buffer[:PACKET_SIZE]

            return {
                "frame_type": frame_type,
                "timestamp_ms": ts_ms,
                "ir_raw": ir_raw,
                "red_raw": red_raw,
                "heuristic_bpm": round(bpm_x10 / 10.0, 1)
            }

        return None
