#!/usr/bin/env python3
"""Deterministic PNG test cards for the omni bank-v2 vision split.

WHY GENERATED CARDS AND NOT PHOTOGRAPHS.  A vision lane is only usable as a
calibration lane if the ground truth is known EXACTLY and cannot drift: "how many
blue squares" has one answer when the image was drawn by this file and an
arguable answer when it was scraped.  Five cards are drawn here, their ground
truth is a by-product of drawing them, and their sha256 goes into the bank so a
later run proves it graded the same pixels.  R6's bank-v2 sketch also allows four
curated photographs; those are a SECOND set for a later cut and nothing here
blocks them.

NO DEPENDENCIES ON PURPOSE.  Pillow is not in `model-surgery-lab/venv` and adding
a package to a venv shared with a running campaign to draw five rectangles would
be a bad trade.  A PNG is a signature, three chunks and a zlib stream, and a 5x7
bitmap font is a table; both are in the standard library's reach and both are
byte-deterministic, which Pillow's encoder settings are not guaranteed to be
across versions.

THE FONT is a plain 5x7 cell font written out as text so it can be read and
checked by eye.  It carries the glyphs the five cards use and nothing else;
`render_text` refuses an unknown character rather than substituting a blank,
because a silently dropped digit would corrupt a card's ground truth.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path
from typing import Any

GLYPH_WIDTH = 5
GLYPH_HEIGHT = 7


# --------------------------------------------------------------------------
# A deflate stream this file writes itself, because zlib's does not reproduce.
# --------------------------------------------------------------------------
#
# THE BUG THIS REPLACES.  The cards used `zlib.compress(raw, 9)` and this file
# claimed they were byte-deterministic.  They are not: rebuilding the same bank
# on the workstation (CPython 3.14) and on the AI server (CPython 3.12) produced
# DIFFERENT bytes for three of the five cards, and therefore a different bank
# digest on each machine -- measured, not suspected.  zlib's output is a property
# of the zlib build, and nothing in its contract pins it.  A bank whose digest
# depends on which machine authored it cannot pin what a review graded.
#
# WHAT THIS DOES INSTEAD.  One deflate block with FIXED Huffman codes (RFC 1951
# section 3.2.6, a table in the specification, identical everywhere) and a match
# finder that is three lines of arithmetic: at each position try a run of the
# previous byte and a repeat of the previous PNG row, take the longer, emit a
# literal when neither reaches three bytes.  Flat rectangles and repeated rows
# are exactly what these cards are made of, so the output stays within a few
# kilobytes of zlib's while being decided entirely by this file.  `zlib.decompress`
# reads it back -- that round trip is asserted per card in the tests.

#: (base length, extra bits) for length codes 257..285.  RFC 1951, table 3.2.5.
_LENGTH_TABLE = [
    (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0),
    (11, 1), (13, 1), (15, 1), (17, 1),
    (19, 2), (23, 2), (27, 2), (31, 2),
    (35, 3), (43, 3), (51, 3), (59, 3),
    (67, 4), (83, 4), (99, 4), (115, 4),
    (131, 5), (163, 5), (195, 5), (227, 5),
    (258, 0),
]

#: (base distance, extra bits) for distance codes 0..29.
_DISTANCE_TABLE = [
    (1, 0), (2, 0), (3, 0), (4, 0),
    (5, 1), (7, 1),
    (9, 2), (13, 2),
    (17, 3), (25, 3),
    (33, 4), (49, 4),
    (65, 5), (97, 5),
    (129, 6), (193, 6),
    (257, 7), (385, 7),
    (513, 8), (769, 8),
    (1025, 9), (1537, 9),
    (2049, 10), (3073, 10),
    (4097, 11), (6145, 11),
    (8193, 12), (12289, 12),
    (16385, 13), (24577, 13),
]

MAXIMUM_MATCH = 258
MINIMUM_MATCH = 3


class _BitWriter:
    """Deflate's two bit orders, kept apart on purpose.

    Huffman codes go out most-significant bit first; everything else goes out
    least-significant bit first.  Mixing them is the classic way to write a
    stream that almost decodes.
    """

    def __init__(self) -> None:
        self.out = bytearray()
        self._accumulator = 0
        self._count = 0

    def raw(self, value: int, width: int) -> None:
        self._accumulator |= (value & ((1 << width) - 1)) << self._count
        self._count += width
        while self._count >= 8:
            self.out.append(self._accumulator & 0xFF)
            self._accumulator >>= 8
            self._count -= 8

    def code(self, value: int, width: int) -> None:
        for shift in range(width - 1, -1, -1):
            self.raw((value >> shift) & 1, 1)

    def flush(self) -> bytes:
        if self._count:
            self.out.append(self._accumulator & 0xFF)
            self._accumulator = 0
            self._count = 0
        return bytes(self.out)


def _fixed_literal(writer: _BitWriter, symbol: int) -> None:
    """The fixed literal/length code of RFC 1951 section 3.2.6."""

    if symbol < 144:
        writer.code(0x30 + symbol, 8)
    elif symbol < 256:
        writer.code(0x190 + symbol - 144, 9)
    elif symbol < 280:
        writer.code(symbol - 256, 7)
    else:
        writer.code(0xC0 + symbol - 280, 8)


def _length_symbol(length: int) -> tuple[int, int, int]:
    for index in range(len(_LENGTH_TABLE) - 1, -1, -1):
        base, extra = _LENGTH_TABLE[index]
        if length >= base:
            return 257 + index, length - base, extra
    raise ValueError(f"length {length} is not encodable")


def _distance_symbol(distance: int) -> tuple[int, int, int]:
    for index in range(len(_DISTANCE_TABLE) - 1, -1, -1):
        base, extra = _DISTANCE_TABLE[index]
        if distance >= base:
            return index, distance - base, extra
    raise ValueError(f"distance {distance} is not encodable")


def _match_length(data: bytes, position: int, distance: int, limit: int) -> int:
    if distance <= 0 or distance > position:
        return 0
    length = 0
    while length < limit and data[position + length] == data[position - distance + length]:
        length += 1
    return length


def deflate_deterministic(data: bytes, *, row_length: int) -> bytes:
    """A zlib stream whose bytes are decided here and nowhere else.

    `row_length` is the PNG scanline stride including its filter byte, offered to
    the matcher as a candidate distance so an unchanged row costs a few bits
    instead of a few hundred bytes.  It is a compression hint only: any value
    produces a valid stream, and the same value always produces the same bytes.
    """

    writer = _BitWriter()
    writer.raw(1, 1)  # BFINAL
    writer.raw(1, 2)  # BTYPE = 01, fixed Huffman codes
    position = 0
    total = len(data)
    candidates = [1, row_length] if row_length and row_length != 1 else [1]
    while position < total:
        limit = min(MAXIMUM_MATCH, total - position)
        best_length = 0
        best_distance = 0
        if limit >= MINIMUM_MATCH:
            for distance in candidates:
                length = _match_length(data, position, distance, limit)
                if length > best_length:
                    best_length, best_distance = length, distance
        if best_length >= MINIMUM_MATCH:
            symbol, extra_value, extra_bits = _length_symbol(best_length)
            _fixed_literal(writer, symbol)
            if extra_bits:
                writer.raw(extra_value, extra_bits)
            dist_symbol, dist_extra, dist_bits = _distance_symbol(best_distance)
            writer.code(dist_symbol, 5)
            if dist_bits:
                writer.raw(dist_extra, dist_bits)
            position += best_length
        else:
            _fixed_literal(writer, data[position])
            position += 1
    _fixed_literal(writer, 256)  # end of block
    body = writer.flush()
    # CMF 0x78 / FLG 0x01: deflate, 32 KiB window, no preset dictionary, and
    # (0x78 << 8 | 0x01) % 31 == 0 as the header check requires.
    return b"\x78\x01" + body + struct.pack(">I", zlib.adler32(data) & 0xFFFFFFFF)

_FONT_SOURCE = {
    "0": ".###.|#...#|#..##|#.#.#|##..#|#...#|.###.",
    "1": "..#..|.##..|..#..|..#..|..#..|..#..|.###.",
    "2": ".###.|#...#|....#|...#.|..#..|.#...|#####",
    "3": "#####|...#.|..#..|...#.|....#|#...#|.###.",
    "4": "...#.|..##.|.#.#.|#..#.|#####|...#.|...#.",
    "5": "#####|#....|####.|....#|....#|#...#|.###.",
    "6": "..##.|.#...|#....|####.|#...#|#...#|.###.",
    "7": "#####|....#|...#.|..#..|.#...|.#...|.#...",
    "8": ".###.|#...#|#...#|.###.|#...#|#...#|.###.",
    "9": ".###.|#...#|#...#|.####|....#|...#.|.##..",
    "A": ".###.|#...#|#...#|#####|#...#|#...#|#...#",
    "B": "####.|#...#|#...#|####.|#...#|#...#|####.",
    "C": ".###.|#...#|#....|#....|#....|#...#|.###.",
    "D": "###..|#..#.|#...#|#...#|#...#|#..#.|###..",
    "E": "#####|#....|#....|####.|#....|#....|#####",
    "F": "#####|#....|#....|####.|#....|#....|#....",
    "G": ".###.|#...#|#....|#.###|#...#|#...#|.###.",
    "H": "#...#|#...#|#...#|#####|#...#|#...#|#...#",
    "I": ".###.|..#..|..#..|..#..|..#..|..#..|.###.",
    "J": "....#|....#|....#|....#|#...#|#...#|.###.",
    "K": "#...#|#..#.|#.#..|##...|#.#..|#..#.|#...#",
    "L": "#....|#....|#....|#....|#....|#....|#####",
    "M": "#...#|##.##|#.#.#|#...#|#...#|#...#|#...#",
    "N": "#...#|##..#|#.#.#|#..##|#...#|#...#|#...#",
    "O": ".###.|#...#|#...#|#...#|#...#|#...#|.###.",
    "P": "####.|#...#|#...#|####.|#....|#....|#....",
    "Q": ".###.|#...#|#...#|#...#|#.#.#|#..#.|.##.#",
    "R": "####.|#...#|#...#|####.|#.#..|#..#.|#...#",
    "S": ".####|#....|#....|.###.|....#|....#|####.",
    "T": "#####|..#..|..#..|..#..|..#..|..#..|..#..",
    "U": "#...#|#...#|#...#|#...#|#...#|#...#|.###.",
    "V": "#...#|#...#|#...#|#...#|#...#|.#.#.|..#..",
    "W": "#...#|#...#|#...#|#.#.#|#.#.#|##.##|#...#",
    "X": "#...#|#...#|.#.#.|..#..|.#.#.|#...#|#...#",
    "Y": "#...#|#...#|.#.#.|..#..|..#..|..#..|..#..",
    "Z": "#####|....#|...#.|..#..|.#...|#....|#####",
    ".": ".....|.....|.....|.....|.....|.##..|.##..",
    ":": ".....|.##..|.##..|.....|.##..|.##..|.....",
    "-": ".....|.....|.....|#####|.....|.....|.....",
    "/": "....#|....#|...#.|..#..|.#...|#....|#....",
    "$": "..#..|.####|#.#..|.###.|..#.#|####.|..#..",
    " ": ".....|.....|.....|.....|.....|.....|.....",
}

FONT: dict[str, tuple[str, ...]] = {}
for _character, _rows in _FONT_SOURCE.items():
    _split = tuple(_rows.split("|"))
    if len(_split) != GLYPH_HEIGHT or any(len(r) != GLYPH_WIDTH for r in _split):
        raise SystemExit(f"REFUSED: glyph {_character!r} is not 5x7.")
    FONT[_character] = _split

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 30, 30)
BLUE = (30, 60, 200)
GREEN = (30, 150, 60)
GREY = (150, 150, 150)


class Canvas:
    """A row-major RGB framebuffer with the four primitives the cards need."""

    def __init__(self, width: int, height: int, background: tuple[int, int, int] = WHITE):
        self.width = int(width)
        self.height = int(height)
        self.pixels = bytearray(bytes(background) * self.width * self.height)

    def set(self, x: int, y: int, colour: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = (y * self.width + x) * 3
            self.pixels[offset : offset + 3] = bytes(colour)

    def rectangle(
        self, x: int, y: int, width: int, height: int, colour: tuple[int, int, int]
    ) -> None:
        for row in range(y, y + height):
            for column in range(x, x + width):
                self.set(column, row, colour)

    def disc(self, cx: int, cy: int, radius: int, colour: tuple[int, int, int]) -> None:
        for row in range(cy - radius, cy + radius + 1):
            for column in range(cx - radius, cx + radius + 1):
                if (column - cx) ** 2 + (row - cy) ** 2 <= radius * radius:
                    self.set(column, row, colour)

    def text(
        self,
        x: int,
        y: int,
        message: str,
        *,
        scale: int = 3,
        colour: tuple[int, int, int] = BLACK,
    ) -> None:
        cursor = x
        for character in message.upper():
            glyph = FONT.get(character)
            if glyph is None:
                raise SystemExit(
                    f"REFUSED: the card font has no glyph for {character!r}; a "
                    "dropped character would corrupt the card's ground truth."
                )
            for row_index, row in enumerate(glyph):
                for column_index, cell in enumerate(row):
                    if cell != "#":
                        continue
                    self.rectangle(
                        cursor + column_index * scale,
                        y + row_index * scale,
                        scale,
                        scale,
                        colour,
                    )
            cursor += (GLYPH_WIDTH + 1) * scale

    def to_png(self) -> bytes:
        raw = bytearray()
        stride = self.width * 3
        for row in range(self.height):
            raw.append(0)  # filter type 0, so the bytes are the pixels
            raw.extend(self.pixels[row * stride : (row + 1) * stride])

        def chunk(tag: bytes, payload: bytes) -> bytes:
            return (
                struct.pack(">I", len(payload))
                + tag
                + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
            )

        header = struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", deflate_deterministic(bytes(raw), row_length=stride + 1))
            + chunk(b"IEND", b"")
        )


# --------------------------------------------------------------------------
# The five cards.  Each returns (canvas, ground truth), and the ground truth is
# produced by the same code that draws the pixels -- it cannot disagree with the
# image because there is no second source for it.
# --------------------------------------------------------------------------


def card_receipt() -> tuple[Canvas, dict[str, Any]]:
    canvas = Canvas(480, 400)
    canvas.rectangle(40, 30, 400, 340, (245, 245, 240))
    canvas.rectangle(40, 30, 400, 4, BLACK)
    canvas.text(70, 55, "MAPLE STREET CAFE", scale=3)
    canvas.text(70, 105, "DATE 2026-03-14", scale=3)
    canvas.text(70, 155, "COFFEE      4.25", scale=3)
    canvas.text(70, 195, "BAGEL       3.50", scale=3)
    canvas.text(70, 235, "TAX         0.62", scale=3)
    canvas.rectangle(70, 275, 340, 3, BLACK)
    canvas.text(70, 295, "TOTAL       8.37", scale=3)
    return canvas, {
        "total": "8.37",
        "date": "2026-03-14",
        "line_items": 3,
        "answerable_from": "the printed TOTAL and DATE lines",
    }


def card_count_shapes() -> tuple[Canvas, dict[str, Any]]:
    canvas = Canvas(480, 400)
    blue_positions = [(60, 60), (200, 60), (340, 60), (130, 200)]
    red_positions = [(60, 320), (270, 200), (410, 200)]
    for x, y in blue_positions:
        canvas.rectangle(x, y, 70, 70, BLUE)
    for x, y in red_positions:
        canvas.rectangle(x, y, 70, 70, RED)
    return canvas, {
        "blue_squares": len(blue_positions),
        "red_squares": len(red_positions),
        "total_squares": len(blue_positions) + len(red_positions),
    }


def card_chart_trend() -> tuple[Canvas, dict[str, Any]]:
    canvas = Canvas(520, 400)
    months = ["JAN", "FEB", "MAR", "APR", "MAY"]
    heights = [60, 95, 140, 195, 260]
    canvas.rectangle(60, 330, 420, 3, BLACK)
    for index, (month, height) in enumerate(zip(months, heights)):
        x = 80 + index * 80
        canvas.rectangle(x, 330 - height, 50, height, BLUE)
        canvas.text(x + 2, 345, month, scale=2)
    return canvas, {
        "months": months,
        "bar_heights": heights,
        "highest_month": "MAY",
        "lowest_month": "JAN",
        "direction": "monotonically increasing",
    }


def card_scene_relation() -> tuple[Canvas, dict[str, Any]]:
    canvas = Canvas(480, 320)
    canvas.disc(130, 160, 70, BLUE)
    canvas.rectangle(290, 90, 140, 140, RED)
    return canvas, {
        "left_shape": "circle",
        "left_colour": "blue",
        "right_shape": "square",
        "right_colour": "red",
    }


def card_grid_select() -> tuple[Canvas, dict[str, Any]]:
    canvas = Canvas(420, 420)
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    green_letter = "E"
    for index, letter in enumerate(letters):
        row, column = divmod(index, 3)
        x, y = 30 + column * 130, 30 + row * 130
        colour = GREEN if letter == green_letter else GREY
        canvas.rectangle(x, y, 110, 110, colour)
        canvas.text(x + 40, y + 35, letter, scale=6, colour=WHITE)
    return canvas, {
        "green_letter": green_letter,
        "cells": letters,
        "grid": "3x3",
    }


CARDS = {
    "vis-ocr-receipt-total-A": card_receipt,
    "vis-count-shapes-B": card_count_shapes,
    "vis-chart-trend-B": card_chart_trend,
    "vis-scene-relation-A": card_scene_relation,
    "vis-grid-cell-select-A": card_grid_select,
}


def build_cards(destination: Path) -> list[dict[str, Any]]:
    """Draw all five cards, write them, and return their asset rows."""

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    assets: list[dict[str, Any]] = []
    for asset_id, builder in CARDS.items():
        canvas, truth = builder()
        payload = canvas.to_png()
        path = destination / f"{asset_id}.png"
        path.write_bytes(payload)
        assets.append(
            {
                "asset_id": asset_id,
                "path": path.name,
                "mime": "image/png",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "width": canvas.width,
                "height": canvas.height,
                "source": "oc_testcards.py, drawn here, no external material",
                "licence": "estate-generated, no third-party rights",
                "ground_truth": truth,
            }
        )
    return assets


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="directory to draw the cards into")
    args = parser.parse_args()
    assets = build_cards(Path(args.out))
    for asset in assets:
        print(
            f"{asset['asset_id']:28s} {asset['width']}x{asset['height']} "
            f"{asset['bytes']:7d} B  {asset['sha256'][:16]}"
        )
    print(json.dumps({"assets": len(assets), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
