"""ROS map coordinates and a conservative reference planner for virtual robots."""
from dataclasses import dataclass, field
from functools import lru_cache
import heapq
import math


def wrap_yaw(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    yaw: float = 0.0

    def packet(self):
        return {"x": self.x, "y": self.y, "yaw": self.yaw}


@dataclass(frozen=True)
class GridMap:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    cells: tuple
    revision: str
    _free_cache: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        if (self.width < 1 or self.height < 1 or self.width * self.height > 262144
                or len(self.cells) != self.width * self.height or not self.revision
                or not all(math.isfinite(v) for v in
                           (self.resolution, self.origin_x, self.origin_y, self.origin_yaw))
                or self.resolution < .01 or any(v < -1 or v > 100 for v in self.cells)):
            raise ValueError("Invalid map")

    def local(self, x, y):
        x, y = x - self.origin_x, y - self.origin_y
        c, s = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        return c * x + s * y, -s * x + c * y

    def world(self, x, y):
        c, s = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        return self.origin_x + c * x - s * y, self.origin_y + s * x + c * y

    def cell(self, x, y):
        return self.cells[y * self.width + x] if 0 <= x < self.width and 0 <= y < self.height else -1

    def centre(self, x, y):
        return self.world((x + .5) * self.resolution, (y + .5) * self.resolution)

    def indices(self, x, y):
        x, y = self.local(x, y)
        return math.floor(x / self.resolution), math.floor(y / self.resolution)

    def valid(self, x, y, radius):
        if not all(math.isfinite(v) for v in (x, y, radius)) or radius < 0:
            return False
        px, py = self.local(x, y)
        cx, cy = self.indices(x, y)
        if not (0 <= self.cell(cx, cy) < 50):
            return False
        if min(px, py) < radius or px > self.width * self.resolution - radius or py > self.height * self.resolution - radius:
            return False
        extent = math.ceil(radius / self.resolution) + 1
        for iy in range(cy - extent, cy + extent + 1):
            for ix in range(cx - extent, cx + extent + 1):
                if 0 <= self.cell(ix, iy) < 50:
                    continue
                dx = max(ix * self.resolution - px, 0, px - (ix + 1) * self.resolution)
                dy = max(iy * self.resolution - py, 0, py - (iy + 1) * self.resolution)
                if dx * dx + dy * dy <= radius * radius:
                    return False
        return True

    def segment_free(self, a, b, radius):
        steps = max(1, math.ceil(math.dist(a, b) / (self.resolution * .2)))
        return all(self.valid(a[0] + (b[0] - a[0]) * i / steps,
                              a[1] + (b[1] - a[1]) * i / steps, radius)
                   for i in range(steps + 1))

    def free_cells(self, radius):
        if radius not in self._free_cache:
            free = frozenset((x, y) for y in range(self.height) for x in range(self.width)
                             if self.valid(*self.centre(x, y), radius))
            if len(self._free_cache) >= 4:
                self._free_cache.clear()
            self._free_cache[radius] = free
        return self._free_cache[radius]

    def plan(self, start, goal, radius):
        if not self.valid(*start, radius) or not self.valid(*goal, radius):
            return []
        first, target = self.indices(*start), self.indices(*goal)
        free = self.free_cells(radius)
        if first not in free or target not in free:
            return []
        heap, costs, parents, closed = [(0, first)], {first: 0}, {}, set()
        while heap:
            _, current = heapq.heappop(heap)
            if current in closed:
                continue
            if current == target:
                reverse = [self.centre(*target)]
                while current in parents:
                    current = parents[current]
                    reverse.append(self.centre(*current))
                points = [start] + list(reversed(reverse)) + [goal]
                path, anchor = [start], 0
                while anchor < len(points) - 1:
                    end = len(points) - 1
                    while end > anchor + 1 and not self.segment_free(points[anchor], points[end], radius):
                        end -= 1
                    if not self.segment_free(points[anchor], points[end], radius):
                        return []
                    path.append(points[end])
                    anchor = end
                return path
            closed.add(current)
            x, y = current
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbor = x + dx, y + dy
                    if (dx == dy == 0 or neighbor not in free or neighbor in closed
                            or (dx and dy and ((x + dx, y) not in free or (x, y + dy) not in free))):
                        continue
                    score = costs[current] + (math.sqrt(2) if dx and dy else 1)
                    if score >= costs.get(neighbor, math.inf):
                        continue
                    costs[neighbor], parents[neighbor] = score, current
                    heapq.heappush(heap, (score + math.dist(neighbor, target), neighbor))
        return []

    def packet(self, session):
        return {"v": 1, "type": "map", "session": session, "map_revision": self.revision,
                "width": self.width, "height": self.height, "resolution": self.resolution,
                "origin_x": self.origin_x, "origin_y": self.origin_y, "origin_yaw": self.origin_yaw,
                "cells": self.cells}


@lru_cache(maxsize=1)
def demo_map():
    w, h, cells = 80, 60, []
    for y in range(h):
        for x in range(w):
            wall = (x < 2 or y < 2 or x >= w - 2 or y >= h - 2
                    or (36 <= x <= 38 and (y < 24 or y > 34))
                    or (13 <= x <= 23 and 25 <= y <= 29)
                    or (51 <= x <= 62 and 14 <= y <= 22)
                    or (54 <= x <= 64 and 42 <= y <= 47))
            cells.append(100 if wall else (-1 if x >= 67 and y >= 46 else 0))
    return GridMap(w, h, .2, -8, -6, 0, tuple(cells), "demo-room-v1")
