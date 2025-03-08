import tkinter as tk
from shapely.geometry import Polygon, box
import numpy as np


class PolygonPacker:
    def __init__(self, root):
        self.root = root
        self.root.title("Polygon Packing with Rectangles")

        # Canvas area
        self.canvas = tk.Canvas(root, bg="white", width=500, height=500)
        self.canvas.grid(row=0, column=0, columnspan=4)

        self.rect_entries = []
        default_sizes = [(40, 20), (20, 20)]

        for i, (w, h) in enumerate(default_sizes):
            tk.Label(root, text=f"Size {i+1} (W, H): ").grid(row=i+1, column=0)
            width_tntry = tk.Entry(root, width=5)
            width_tntry.grid(row=i+1, column=1)
            width_tntry.insert(0, str(w))

            height_entry = tk.Entry(root, width=5)
            height_entry.grid(row=i+1, column=2)
            height_entry.insert(0, str(h))

            self.rect_entries.append((width_tntry, height_entry))

        # Update Buttons
        self.update_button = tk.Button(root, text="Update", command=self.arrange_rectangles)
        self.update_button.grid(row=4, column=1)

        # Clear Button
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=4, column=2)

        # Polygon drawer
        self.points = []  # List of vertices
        self.polygon = None  # Polygon object
        self.rectangles = []  # List of rectangles

        # Mouse event
        self.canvas.bind("<Button-1>", self.add_point)  # Left click to add a vertex
        self.canvas.bind("<Double-Button-1>", self.complete_polygon)  # Double click to complete the polygon
        self.canvas.bind("<Button-3>", self.clear_canvas)   # Right click to reset the canvas

    def get_rectangle_sizes(self):
        """ Get rectangle sizes from the entries """
        sizes = []
        for width_entry, height_entry in self.rect_entries:
            try:
                w = int(width_entry.get())
                h = int(height_entry.get())
                if w > 0 and h > 0:
                    sizes.append((w, h))
            except ValueError:
                continue
        # Sort by area in descending order
        sorted_sizes = sorted(sizes, key=lambda s: s[0] * s[1], reverse=True)
        return sorted_sizes

    def add_point(self, event):
        """ Add a vertex by left click """
        x, y = event.x, event.y
        self.points.append((x, y))
        self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="black", outline="black")

        if len(self.points) > 1:
            self.canvas.create_line(self.points[-2], self.points[-1], fill="black", width=2)

    def complete_polygon(self, event):
        """ Complete the polygon by double click """
        if len(self.points) > 2:
            self.canvas.create_line(self.points[-1], self.points[0], fill="black", width=2)
            self.polygon = Polygon(self.points)  # Create a polygon object
            self.points = []  # Reset the list of vertices
            self.arrange_rectangles()

    def arrange_rectangles(self):
        """ Arrange rectangles to fill the polygon """
        if self.polygon is None:
            return

        # Remove existing rectangles
        for rect_id in self.rectangles:
            self.canvas.delete(rect_id)
        self.rectangles.clear()

        rect_sizes = self.get_rectangle_sizes()
        if not rect_sizes:
            return

        min_x, min_y, max_x, max_y = self.polygon.bounds
        start_x, start_y = int(min_x), int(min_y)

        occupied = set()  # Record used coordinates

        def can_place(x, y, w, h):
            """ Check if a rectangle can be placed at the given position """
            if (x + w > max_x) or (y + h > max_y):
                return False
            rect = box(x, y, x + w, y + h)
            return self.polygon.contains(rect) and not any((x + dx, y + dy) in occupied for dx in range(w) for dy in range(h))

        def place_rectangle(x, y, w, h):
            """ Place rectangles and Record occupied coordinates """
            rect_id = self.canvas.create_rectangle(x, y, x + w, y + h, outline="blue", width=2)
            self.rectangles.append(rect_id)
            for dx in range(w):
                for dy in range(h):
                    occupied.add((x + dx, y + dy))

        # Algorithm to fill the polygon
        step_x = max(10, int(min(rect_sizes, key=lambda s: s[0])[0] / 2))
        step_y = min(rect_sizes, key=lambda s: s[1])[1]
        y = start_y
        while y + min(rect_sizes, key=lambda s: s[1])[1] <= max_y:
            x = start_x
            while x + min(rect_sizes, key=lambda s: s[0])[0] <= max_x:
                placed = False
                for w, h in rect_sizes:  # Try from bigger rectangles
                    if can_place(x, y, w, h):
                        place_rectangle(x, y, w, h)
                        placed = True
                        x += w
                        break
                # x += min(rect_sizes, key=lambda s: s[0])[0]
                if not placed:  # If no rectangle was placed, move by a small fixed step
                    x += step_x
            # y += min(rect_sizes, key=lambda s: s[1])[1]
            y += step_y

    def clear_canvas(self, event=None):
        """ Reset the canvas by right click """
        self.canvas.delete("all")  # Delete all items
        self.points = []  # Reset the list of vertices
        self.polygon = None
        self.rectangles = []


if __name__ == "__main__":
    root = tk.Tk()
    app = PolygonPacker(root)
    root.mainloop()
