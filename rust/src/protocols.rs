use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use crate::modulators::Xorshift64;

// ---------------------------------------------------------------------------
// CRC helpers (internal, not exposed to Python)
// ---------------------------------------------------------------------------

/// CRC-24 for ADS-B.  Polynomial: x^24 + x^23 + x^10 + x^3 + 1  (0x1FFF409).
fn crc24_adsb(data: &[u8], num_bits: usize) -> u32 {
    let poly: u32 = 0x1FFF409;
    let mut crc: u32 = 0;
    for i in 0..num_bits {
        let byte_idx = i / 8;
        let bit_idx = 7 - (i % 8);
        let bit = ((data[byte_idx] >> bit_idx) & 1) as u32;
        crc ^= bit << 23;
        if crc & 0x800000 != 0 {
            crc = (crc << 1) ^ poly;
        } else {
            crc <<= 1;
        }
        crc &= 0xFFFFFF;
    }
    crc
}

/// CRC-16 CCITT (polynomial 0x1021, init 0xFFFF).
fn crc16_ccitt(data: &[u8]) -> u16 {
    let poly: u16 = 0x1021;
    let mut crc: u16 = 0xFFFF;
    for &byte in data {
        crc ^= (byte as u16) << 8;
        for _ in 0..8 {
            if crc & 0x8000 != 0 {
                crc = (crc << 1) ^ poly;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

// ---------------------------------------------------------------------------
// ADS-B frame generator
// ---------------------------------------------------------------------------

/// Generate a random 112-bit (14 byte) ADS-B extended squitter frame.
///
/// Layout:
/// - Bits 0-4:   Downlink Format = 17 (10001)
/// - Bits 5-7:   Capability (random)
/// - Bits 8-31:  24-bit ICAO address (random)
/// - Bits 32-87: 56-bit message data (random)
/// - Bits 88-111: 24-bit CRC over bits 0-87
#[pyfunction]
pub fn generate_adsb_frame<'py>(py: Python<'py>, seed: u64) -> Bound<'py, PyArray1<u8>> {
    let mut rng = Xorshift64::new(seed);

    let mut frame = [0u8; 14];

    // Fill bytes 0..11 with random data first
    for byte in frame[0..11].iter_mut() {
        *byte = (rng.next() & 0xFF) as u8;
    }

    // Set DF=17 in bits 0-4: first 5 bits = 10001
    // byte 0 upper 5 bits = 10001 = 0x88, keep lower 3 bits random (capability)
    frame[0] = 0x88 | (frame[0] & 0x07);

    // Compute CRC-24 over bits 0-87 (bytes 0-10, 88 bits)
    let crc = crc24_adsb(&frame[0..11], 88);
    frame[11] = ((crc >> 16) & 0xFF) as u8;
    frame[12] = ((crc >> 8) & 0xFF) as u8;
    frame[13] = (crc & 0xFF) as u8;

    Array1::from(frame.to_vec()).into_pyarray(py)
}

// ---------------------------------------------------------------------------
// AIS frame generator
// ---------------------------------------------------------------------------

/// Perform HDLC bit stuffing: after 5 consecutive 1-bits, insert a 0.
fn hdlc_bit_stuff(bits: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bits.len() + bits.len() / 5);
    let mut ones_count = 0u32;
    for &b in bits {
        out.push(b);
        if b == 1 {
            ones_count += 1;
            if ones_count == 5 {
                out.push(0);
                ones_count = 0;
            }
        } else {
            ones_count = 0;
        }
    }
    out
}

/// Convert a byte slice to a vector of bits (MSB first).
fn bytes_to_bits(data: &[u8], num_bits: usize) -> Vec<u8> {
    let mut bits = Vec::with_capacity(num_bits);
    for i in 0..num_bits {
        let byte_idx = i / 8;
        let bit_idx = 7 - (i % 8);
        bits.push((data[byte_idx] >> bit_idx) & 1);
    }
    bits
}

/// Convert bits to bytes (MSB first), padding the last byte with zeros.
fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    let num_bytes = bits.len().div_ceil(8);
    let mut bytes = vec![0u8; num_bytes];
    for (i, &b) in bits.iter().enumerate() {
        if b == 1 {
            bytes[i / 8] |= 1 << (7 - (i % 8));
        }
    }
    bytes
}

/// Generate a random 168-bit AIS message type 1 with HDLC framing.
///
/// Layout:
/// - Bits 0-5:   Message type = 1 (000001)
/// - Bits 6-7:   Repeat indicator = 0 (00)
/// - Bits 8-37:  30-bit MMSI (random)
/// - Bits 38-167: Random navigational data
/// - CRC-16 CCITT appended
/// - HDLC framed: 0x7E | bit-stuffed payload | 0x7E
#[pyfunction]
pub fn generate_ais_frame<'py>(py: Python<'py>, seed: u64) -> Bound<'py, PyArray1<u8>> {
    let mut rng = Xorshift64::new(seed);

    // Build 168 bits of message data as bytes (21 bytes)
    let mut msg_bytes = [0u8; 21];
    for byte in msg_bytes.iter_mut() {
        *byte = (rng.next() & 0xFF) as u8;
    }

    // Set message type = 1 (000001) in bits 0-5
    // Byte 0: bits 0-5 = 000001, bits 6-7 = 00 (repeat indicator)
    // So byte 0 = 0000_0100 = 0x04
    msg_bytes[0] = 0x04;

    // Compute CRC-16 over the 21 message bytes
    let crc = crc16_ccitt(&msg_bytes);
    let crc_bytes = [(crc >> 8) as u8, (crc & 0xFF) as u8];

    // Convert message + CRC to bits
    let mut all_bits = bytes_to_bits(&msg_bytes, 168);
    all_bits.extend_from_slice(&bytes_to_bits(&crc_bytes, 16));

    // Bit stuffing
    let stuffed = hdlc_bit_stuff(&all_bits);

    // Build final frame: flag + stuffed bits + flag
    let flag_bits: [u8; 8] = [0, 1, 1, 1, 1, 1, 1, 0]; // 0x7E
    let mut frame_bits = Vec::with_capacity(8 + stuffed.len() + 8);
    frame_bits.extend_from_slice(&flag_bits);
    frame_bits.extend_from_slice(&stuffed);
    frame_bits.extend_from_slice(&flag_bits);

    // Convert to bytes
    let frame_bytes = bits_to_bytes(&frame_bits);
    Array1::from(frame_bytes).into_pyarray(py)
}

// ---------------------------------------------------------------------------
// ACARS frame generator
// ---------------------------------------------------------------------------

/// Add odd parity bit (MSB) to a 7-bit character value.
fn add_odd_parity(ch: u8) -> u8 {
    let data = ch & 0x7F;
    let ones = data.count_ones();
    if ones % 2 == 0 {
        data | 0x80 // set parity bit to make odd
    } else {
        data
    }
}

/// Generate a random ACARS message frame.
///
/// Structure:
/// - Preamble: 0xAA, 0xAA (16 alternating bits)
/// - Sync: 0x2B ('+'), 0x2A ('*')
/// - Mode character (random ASCII letter, odd parity)
/// - Address: 7 random alphanumeric characters (odd parity each)
/// - Message: 10-20 random printable ASCII characters (odd parity each)
/// - ETX: 0x03
/// - BCS: CRC-16 over message portion
#[pyfunction]
pub fn generate_acars_frame<'py>(py: Python<'py>, seed: u64) -> Bound<'py, PyArray1<u8>> {
    let mut rng = Xorshift64::new(seed);

    let mut frame = Vec::with_capacity(40);

    // Preamble
    frame.push(0xAA);
    frame.push(0xAA);

    // Sync characters
    frame.push(0x2B); // '+'
    frame.push(0x2A); // '*'

    // Track start of CRC region (after sync)
    let crc_start = frame.len();

    // Mode character: random ASCII uppercase letter A-Z
    let mode = b'A' + (rng.next() % 26) as u8;
    frame.push(add_odd_parity(mode));

    // Address: 7 random alphanumeric characters
    let alphanum = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    for _ in 0..7 {
        let ch = alphanum[(rng.next() as usize) % alphanum.len()];
        frame.push(add_odd_parity(ch));
    }

    // Message: 10-20 random printable ASCII characters
    let msg_len = 10 + (rng.next() % 11) as usize;
    for _ in 0..msg_len {
        // Printable ASCII: 0x20 - 0x7E
        let ch = 0x20 + (rng.next() % 95) as u8;
        frame.push(add_odd_parity(ch));
    }

    // ETX
    frame.push(0x03);

    // CRC-16 over the payload (from mode char through ETX)
    let crc = crc16_ccitt(&frame[crc_start..]);
    frame.push((crc >> 8) as u8);
    frame.push((crc & 0xFF) as u8);

    Array1::from(frame).into_pyarray(py)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_adsb_frame(seed: u64) -> Vec<u8> {
        let mut rng = Xorshift64::new(seed);
        let mut frame = [0u8; 14];
        for byte in frame[0..11].iter_mut() {
            *byte = (rng.next() & 0xFF) as u8;
        }
        frame[0] = 0x88 | (frame[0] & 0x07);
        let crc = crc24_adsb(&frame[0..11], 88);
        frame[11] = ((crc >> 16) & 0xFF) as u8;
        frame[12] = ((crc >> 8) & 0xFF) as u8;
        frame[13] = (crc & 0xFF) as u8;
        frame.to_vec()
    }

    fn build_ais_frame(seed: u64) -> Vec<u8> {
        let mut rng = Xorshift64::new(seed);
        let mut msg_bytes = [0u8; 21];
        for byte in msg_bytes.iter_mut() {
            *byte = (rng.next() & 0xFF) as u8;
        }
        msg_bytes[0] = 0x04;
        let crc = crc16_ccitt(&msg_bytes);
        let crc_bytes = [(crc >> 8) as u8, (crc & 0xFF) as u8];
        let mut all_bits = bytes_to_bits(&msg_bytes, 168);
        all_bits.extend_from_slice(&bytes_to_bits(&crc_bytes, 16));
        let stuffed = hdlc_bit_stuff(&all_bits);
        let flag_bits: [u8; 8] = [0, 1, 1, 1, 1, 1, 1, 0];
        let mut frame_bits = Vec::with_capacity(8 + stuffed.len() + 8);
        frame_bits.extend_from_slice(&flag_bits);
        frame_bits.extend_from_slice(&stuffed);
        frame_bits.extend_from_slice(&flag_bits);
        bits_to_bytes(&frame_bits)
    }

    fn build_acars_frame(seed: u64) -> Vec<u8> {
        let mut rng = Xorshift64::new(seed);
        let mut frame = Vec::with_capacity(40);
        frame.push(0xAA);
        frame.push(0xAA);
        frame.push(0x2B);
        frame.push(0x2A);
        let crc_start = frame.len();
        let mode = b'A' + (rng.next() % 26) as u8;
        frame.push(add_odd_parity(mode));
        let alphanum = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        for _ in 0..7 {
            let ch = alphanum[(rng.next() as usize) % alphanum.len()];
            frame.push(add_odd_parity(ch));
        }
        let msg_len = 10 + (rng.next() % 11) as usize;
        for _ in 0..msg_len {
            let ch = 0x20 + (rng.next() % 95) as u8;
            frame.push(add_odd_parity(ch));
        }
        frame.push(0x03);
        let crc = crc16_ccitt(&frame[crc_start..]);
        frame.push((crc >> 8) as u8);
        frame.push((crc & 0xFF) as u8);
        frame
    }

    #[test]
    fn adsb_frame_length() {
        let frame = build_adsb_frame(42);
        assert_eq!(frame.len(), 14, "ADS-B frame must be 14 bytes");
    }

    #[test]
    fn adsb_df17() {
        let frame = build_adsb_frame(42);
        assert_eq!(frame[0] >> 3, 17, "DF field must be 17");
    }

    #[test]
    fn adsb_crc_verification() {
        let data = build_adsb_frame(123);
        let crc = crc24_adsb(&data[0..11], 88);
        let embedded_crc = ((data[11] as u32) << 16) | ((data[12] as u32) << 8) | (data[13] as u32);
        assert_eq!(crc, embedded_crc, "CRC-24 must match");
    }

    #[test]
    fn ais_frame_contains_flags() {
        let data = build_ais_frame(42);
        assert_eq!(data[0], 0x7E, "AIS frame must start with 0x7E flag");
        assert!(data.len() > 2, "AIS frame must be longer than 2 bytes");
    }

    #[test]
    fn acars_frame_preamble_and_sync() {
        let data = build_acars_frame(42);
        assert_eq!(data[0], 0xAA, "ACARS preamble byte 0");
        assert_eq!(data[1], 0xAA, "ACARS preamble byte 1");
        assert_eq!(data[2], 0x2B, "ACARS sync '+'");
        assert_eq!(data[3], 0x2A, "ACARS sync '*'");
        assert!(data.len() >= 25, "ACARS frame too short: {}", data.len());
    }

    #[test]
    fn adsb_deterministic() {
        let f1 = build_adsb_frame(99);
        let f2 = build_adsb_frame(99);
        assert_eq!(f1, f2);
    }

    #[test]
    fn crc24_fits_24_bits() {
        // CRC should always fit in 24 bits
        let data = [0xFFu8; 11];
        let crc = crc24_adsb(&data, 88);
        assert!(crc <= 0xFFFFFF, "CRC-24 must fit in 24 bits");
        assert_ne!(crc, 0, "CRC of all-ones should be non-zero");
    }

    #[test]
    fn crc16_known_value() {
        let data = [0u8; 4];
        let crc = crc16_ccitt(&data);
        assert_ne!(
            crc, 0,
            "CRC of all zeros should be non-zero with 0xFFFF init"
        );
    }

    #[test]
    fn hdlc_bit_stuffing() {
        // 5 ones should get a 0 inserted
        let bits = vec![1, 1, 1, 1, 1, 0];
        let stuffed = hdlc_bit_stuff(&bits);
        assert_eq!(stuffed, vec![1, 1, 1, 1, 1, 0, 0]);
    }

    #[test]
    fn odd_parity() {
        // 'A' = 0x41 = 0100_0001, two 1-bits (even), so parity bit set
        assert_eq!(add_odd_parity(b'A'), b'A' | 0x80);
        // 'C' = 0x43 = 0100_0011, three 1-bits (odd), no parity bit
        assert_eq!(add_odd_parity(b'C'), b'C');
    }
}
