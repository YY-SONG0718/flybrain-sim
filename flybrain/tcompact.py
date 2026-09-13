"""Decoder for the Apache Thrift *compact* binary protocol.

Just enough to read Parquet file metadata and page headers without pyarrow.
Every reader takes the byte buffer and a position and returns (value, new_position).
"""
from __future__ import annotations

import struct
from typing import Any

# Thrift compact-protocol field type codes
TYPE_BOOL_TRUE = 1
TYPE_BOOL_FALSE = 2
TYPE_BYTE = 3
TYPE_I16 = 4
TYPE_I32 = 5
TYPE_I64 = 6
TYPE_DOUBLE = 7
TYPE_BINARY = 8
TYPE_LIST = 9
TYPE_SET = 10
TYPE_MAP = 11
TYPE_STRUCT = 12

ThriftStruct = dict[int, Any]


def read_varint(buffer: bytes, position: int) -> tuple[int, int]:
    """
    Read an unsigned LEB128 varint.

    :param buffer: Bytes to read from.
    :param position: Offset of the first byte.
    :return: (value, position after the varint).
    """
    result = 0
    shift = 0
    while True:
        byte = buffer[position]
        position += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, position
        shift += 7


def zigzag_decode(value: int) -> int:
    """
    Undo Thrift's zigzag encoding of a signed integer.

    :param value: Zigzag-encoded unsigned integer.
    :return: Signed integer.
    """
    return (value >> 1) ^ -(value & 1)


def read_struct(buffer: bytes, position: int) -> tuple[ThriftStruct, int]:
    """
    Read one struct as a dict of field id -> value.

    :param buffer: Bytes to read from.
    :param position: Offset of the first field header.
    :return: (fields, position after the terminating 0 byte).
    """
    fields: ThriftStruct = {}
    last_field_id = 0
    while True:
        header = buffer[position]
        position += 1
        if header == 0:
            return fields, position
        field_id_delta = (header >> 4) & 0xF
        field_type = header & 0xF
        if field_id_delta == 0:
            zigzagged, position = read_varint(buffer, position)
            field_id = zigzag_decode(zigzagged)
        else:
            field_id = last_field_id + field_id_delta
        last_field_id = field_id
        if field_type == TYPE_BOOL_TRUE:
            fields[field_id] = True
        elif field_type == TYPE_BOOL_FALSE:
            fields[field_id] = False
        else:
            fields[field_id], position = read_value(buffer, position, field_type)


def read_value(buffer: bytes, position: int, field_type: int) -> tuple[Any, int]:
    """
    Read one value of the given compact-protocol type.

    :param buffer: Bytes to read from.
    :param position: Offset of the value.
    :param field_type: Thrift compact type code.
    :return: (value, position after the value).
    :raises ValueError: if the type code is unknown.
    """
    if field_type == TYPE_BOOL_TRUE:
        return True, position
    if field_type == TYPE_BOOL_FALSE:
        return False, position
    if field_type == TYPE_BYTE:
        return struct.unpack("b", buffer[position:position + 1])[0], position + 1
    if field_type in (TYPE_I16, TYPE_I32, TYPE_I64):
        zigzagged, position = read_varint(buffer, position)
        return zigzag_decode(zigzagged), position
    if field_type == TYPE_DOUBLE:
        return struct.unpack("<d", buffer[position:position + 8])[0], position + 8
    if field_type == TYPE_BINARY:
        length, position = read_varint(buffer, position)
        return buffer[position:position + length], position + length
    if field_type in (TYPE_LIST, TYPE_SET):
        return _read_list(buffer, position)
    if field_type == TYPE_MAP:
        return _read_map(buffer, position)
    if field_type == TYPE_STRUCT:
        return read_struct(buffer, position)
    raise ValueError(f"unknown thrift compact type {field_type} at byte {position}")


def _read_list(buffer: bytes, position: int) -> tuple[list[Any], int]:
    """
    Read a list or set.

    :param buffer: Bytes to read from.
    :param position: Offset of the list header.
    :return: (items, position after the list).
    """
    header = buffer[position]
    position += 1
    size = (header >> 4) & 0xF
    element_type = header & 0xF
    if size == 15:
        size, position = read_varint(buffer, position)
    items: list[Any] = []
    for _ in range(size):
        item, position = read_value(buffer, position, element_type)
        items.append(item)
    return items, position


def _read_map(buffer: bytes, position: int) -> tuple[dict[Any, Any], int]:
    """
    Read a map.

    :param buffer: Bytes to read from.
    :param position: Offset of the map size varint.
    :return: (mapping, position after the map).
    """
    size, position = read_varint(buffer, position)
    if size == 0:
        return {}, position
    kv_types = buffer[position]
    position += 1
    key_type = (kv_types >> 4) & 0xF
    value_type = kv_types & 0xF
    mapping: dict[Any, Any] = {}
    for _ in range(size):
        key, position = read_value(buffer, position, key_type)
        value, position = read_value(buffer, position, value_type)
        mapping[key] = value
    return mapping, position
