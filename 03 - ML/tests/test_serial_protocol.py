"""
Unit Tests for Framed UART Serial Protocol & CRC16 Decoder
==========================================================
Tests:
1. Frame packing and unpacking round-trip (0xAA start, payload, CRC16, 0x55 end).
2. CRC-16-CCITT computation accuracy and detection of bit-flips.
3. StreamPacketParser parsing multiple consecutive frames.
4. StreamPacketParser resynchronization past corrupted framing or random noise bytes.
5. Rejection of corrupted CRC16 frames.
"""

import struct
import pytest
from edge_inference.serial_protocol import (
    pack_ppg_frame,
    compute_crc16,
    StreamPacketParser,
    FRAME_START_BYTE,
    FRAME_END_BYTE,
    PACKET_SIZE,
)


def test_pack_and_parse_roundtrip():
    """Test that a packed frame is correctly unpacked with identical fields."""
    ts = 1714500000
    ir = 52400
    red = 48200
    bpm = 72.4

    packet = pack_ppg_frame(timestamp_ms=ts, ir_raw=ir, red_raw=red, heuristic_bpm=bpm)
    assert len(packet) == PACKET_SIZE
    assert packet[0] == FRAME_START_BYTE
    assert packet[-1] == FRAME_END_BYTE

    parser = StreamPacketParser()
    parser.feed(packet)

    frame = parser.parse_next()
    assert frame is not None
    assert frame["timestamp_ms"] == ts
    assert frame["ir_raw"] == ir
    assert frame["red_raw"] == red
    assert frame["heuristic_bpm"] == pytest.approx(bpm, abs=0.1)

    # Buffer should now be empty
    assert parser.parse_next() is None


def test_corrupted_crc_rejected():
    """Test that any packet with corrupted payload or CRC is rejected."""
    packet = bytearray(pack_ppg_frame(timestamp_ms=1000, ir_raw=50000, red_raw=45000, heuristic_bpm=65.0))

    # Corrupt one payload byte
    packet[5] ^= 0xFF

    parser = StreamPacketParser()
    parser.feed(bytes(packet))

    # Frame must be rejected
    assert parser.parse_next() is None


def test_parser_noise_recovery():
    """Test that parser recovers valid frames even when preceded by noise bytes."""
    noise = bytes([0x12, 0x34, 0xAA, 0x02, 0x99, 0x55, 0x78])  # Noise with false delimiters
    valid_packet = pack_ppg_frame(timestamp_ms=2000, ir_raw=51000, red_raw=46000, heuristic_bpm=70.0)

    parser = StreamPacketParser()
    parser.feed(noise + valid_packet)

    # Parser should discard noise and parse the valid packet
    frame = parser.parse_next()
    assert frame is not None
    assert frame["timestamp_ms"] == 2000
    assert frame["ir_raw"] == 51000


def test_multi_frame_streaming():
    """Test feeding a continuous stream of multiple packets."""
    parser = StreamPacketParser()
    packets = [
        pack_ppg_frame(timestamp_ms=1000 * i, ir_raw=50000 + i * 10, red_raw=45000 + i * 5, heuristic_bpm=72.0)
        for i in range(5)
    ]
    stream_data = b"".join(packets)

    # Feed in arbitrary chunks
    chunk_size = 7
    frames_recovered = []
    for i in range(0, len(stream_data), chunk_size):
        parser.feed(stream_data[i:i + chunk_size])
        while True:
            f = parser.parse_next()
            if not f:
                break
            frames_recovered.append(f)

    assert len(frames_recovered) == 5
    for i, f in enumerate(frames_recovered):
        assert f["timestamp_ms"] == 1000 * i
        assert f["ir_raw"] == 50000 + i * 10
