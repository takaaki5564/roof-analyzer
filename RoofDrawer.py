import tkinter as tk
from tkinter import messagebox
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union, polygonize
import random


class RoofDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("Roof Drawer")

        # Main canvas
        self.canvas = tk.Canvas(self.root, width=600, height=600, bg='white')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.log_window = tk.Text(root, height=10, width=50)
        self.log_window.pack(side=tk.RIGHT, fill=tk.Y)

        # Initial state
        self.step = "outline"   # "outline" -> "ridge" -> "valley" -> "split" -> "angle_input"
        self.outline = []       # List of points that define the outline of the roof
        self.ridges = []        # List of points that define the ridge of the roof
        self.valleys = []       # List of points that define the valleys of the roof
        self.sub_roofs = []     # List of sub roofs after splitting
        self.current_roof = None
        self.angle_data = {}  # Dictionary of angles for each sub roof

        self.temp_line = None   # Temporary line for drawing

        # Event bindings
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        self.canvas.bind("<Button-3>", self.on_right_click)

        self.log("ステップ1: 外形を描画してください (左クリックで頂点追加、ダブルクリックで閉じる)")

    def log(self, text):
        self.log_window.insert(tk.END, text + "\n")
        self.log_window.see(tk.END)

    def on_left_click(self, event):
        """ Left click to add a point to the outline """
        x, y = event.x, event.y
        print(f"Left click at ({x}, {y})")

        if self.step == "outline":
            self.outline.append((x, y))
            self.canvas.create_oval(x-2, y-2, x+2, y+2, fill='black')
            if len(self.outline) > 1:
                self.canvas.create_line(self.outline[-2], self.outline[-1], width=2)

        elif self.step in ["ridge", "valley"]:
            if self.temp_line is None:
                self.temp_line = [(x, y)]
                self.canvas.create_oval(x-2, y-2, x+2, y+2, fill='red' if self.step == "ridge" else "green")
            else:
                self.temp_line.append((x, y))
                self.canvas.create_oval(x-2, y-2, x+2, y+2, fill='red', width=2)
                line_id = self.canvas.create_line(self.temp_line, fill="red" if self.step == "ridge" else "green", width=2)
                if self.step == "ridge":
                    self.ridges.append((self.temp_line, line_id))
                else:
                    self.valleys.append((self.temp_line, line_id))
                self.temp_line = None  # Reset for next line

    def on_double_click(self, event):
        """ Double click to finish drawing the outline """
        if self.step in ["ridge" or "valley"] and self.temp_line:
            self.temp_line = None
            self.log(f"{self.step}の描画を終了しました。右クリックで次のステップへ")

    def on_right_click(self, event):
        """ Right click to proceed to the next step """
        if self.step == "outline":
            if len(self.outline) < 3:
                messagebox.showerror("エラー", "3つ以上の頂点が必要です")
                return
            # Close the polygon
            self.canvas.create_line(self.outline[-1], self.outline[0], fill="black", width=2)
            self.log("ステップ2: 屋根の稜線を描画してください (左クリックで頂点追加、右クリックで次のステップへ)")
            self.step = "ridge"

        elif self.step == "ridge":
            if len(self.outline) < 2:
                messagebox.showerror("エラー", "2つ以上の頂点が必要です")
                return
            self.log("ステップ3: 屋根の谷を描画してください (左クリックで頂点追加、右クリックで次のステップへ)")
            self.step = "valley"

        elif self.step == "valley":
            self.log("ステップ4: 屋根の分割処理を実施します")
            self.split_roofs()
            self.step = "split"

        elif self.step == "split":
            self.log("ステップ5: 各部分屋根の角度を入力してください")
            self.step = "angle_input"
            self.current_roof = 0
            self.prompt_angle_input()

        elif self.step == "angle_input":
            self.clear_canvas()

    def split_roofs(self):
        """ Split the roof into sub roofs """

        def find_nearest_point(target, points, threshold=40):
            nearest = None
            min_dist = float('inf')
            for p in points:
                dist = Point(target).distance(Point(p))
                if dist < min_dist and dist <= threshold:
                    min_dist = dist
                    nearest = p
            return nearest

        self.log("近い点を統合します...")

        # Get outlines, ridges, and valleys
        outline_points = set(self.outline)
        ridge_points = set(pt for line, _ in self.ridges for pt in line)
        # valley_points = set(pt for line, _ in self.valleys for pt in line)

        # Merge valley points and lines to the outline or ridge points
        new_valleys = []
        for valley in self.valleys:
            new_line = []
            for pt in valley[0]:
                nearest = find_nearest_point(pt, outline_points.union(ridge_points))
                if nearest:
                    new_line.append(nearest)
                    print(f"Merge {pt} to {nearest}")
                else:
                    new_line.append(pt)

            if len(new_line) >= 2:
                new_valleys.append((new_line, LineString(new_line)))

        self.valleys = new_valleys

        # Clear the canvas and draw the merged lines
        self.canvas.delete("all")
        # print(f"Len outline: {len(self.outline)}")
        # print(f"Len ridges: {len(self.ridges)}")
        # print(f"Len valleys: {len(self.valleys)}")

        # Draw outline
        self.canvas.create_polygon(self.outline, fill="white", outline="black", width=2)
        for pt in self.outline:
            self.canvas.create_oval(pt[0]-2, pt[1]-2, pt[0]+2, pt[1]+2, fill='black')       

        # Draw ridges
        for ridge in self.ridges:
            self.canvas.create_line(ridge[0], fill="red", width=2)
            for pt in ridge[0]:
                self.canvas.create_oval(pt[0]-2, pt[1]-2, pt[0]+2, pt[1]+2, fill='red')

        for valley in self.valleys:
            self.canvas.create_line(valley[0], fill="green", width=2)
            for pt in valley[0]:
                self.canvas.create_oval(pt[0]-2, pt[1]-2, pt[0]+2, pt[1]+2, fill='green')

        self.log("屋根を分割します...")

        # Get outlines
        outline_lines = [LineString([self.outline[i], self.outline[i+1]]) for i in range(len(self.outline)-1)]
        outline_lines.append(LineString([self.outline[-1], self.outline[0]]))  # Close the polygon

        # Get ridges and valleys
        ridge_lines = [LineString(ridge[0]) for ridge in self.ridges]
        valley_lines = [LineString(valley[0]) for valley in self.valleys]

        # Merge all lines
        all_lines = outline_lines + ridge_lines + valley_lines

        # Get intersection of lines and find polygons
        merged_lines = unary_union(all_lines)
        polygons = list(polygonize(merged_lines))

        main_polygon = Polygon(self.outline)

        # Get polygons inside the main polygon
        split_polygons = [poly for poly in polygons if poly.is_valid and poly.within(main_polygon)]

        self.sub_roofs = split_polygons

        self.log(f"分割された部分屋根の数: {len(split_polygons)}")

        # Draw sub roofs
        self.draw_split_polygons()

    def draw_split_polygons(self, outline_color="black"):
        for poly in self.sub_roofs:
            r, g, b = [random.randint(180, 255) for _ in range(3)]
            color = f"#{r:02x}{g:02x}{b:02x}"

            self.canvas.create_polygon(
                [coord for pt in poly.exterior.coords for coord in pt],
                fill=color, outline=outline_color, width=2
            )

    def draw_polygon(self, polygon, fill="", outline_color="black", width=2):
        """ Draw a polygon on the canvas """
        points = list(polygon.exterior.coords)
        poly_id = self.canvas.create_polygon(points, fill=fill, outline=outline_color, width=width)
        return poly_id

    def prompt_angle_input(self):
        """ Prompt the user to input the angle of the current sub roof """
        if self.current_roof >= len(self.sub_roofs):
            self.log("全ての部分屋根の角度を入力しました。終了します")
            return

        # Draw all sub roofs with black color lines
        self.draw_split_polygons()

        # Draw the current sub roof with a red line
        current_poly = self.sub_roofs[self.current_roof]
        self.draw_polygon(current_poly, fill="", outline_color="red", width=3)

        # Input angle by the user
        self.log(f"{self.current_roof+1}番目の部分屋根の角度を入力してください")

        self.angle_data[self.current_roof] = tk.StringVar()

        entry = tk.Entry(self.root, textvariable=self.angle_data[self.current_roof])
        entry.pack()
        button = tk.Button(self.root, text="次へ", command=self.on_angle_input)
        button.pack()

    def on_angle_input(self):
        """ Process the angle input and proceed to the next sub roof """
        angle = self.angle_data[self.current_roof].get()
        if not angle:
            messagebox.showerror("エラー", "角度を入力してください")
            return

        angle = int(angle)
        current_poly = self.sub_roofs[self.current_roof]
        centroid = current_poly.centroid

        self.canvas.create_text(
            centroid.x, centroid.y, text=f"{angle}°", font=("Arial", 12, "bold"), fill="blue"
        )

        self.current_roof += 1
        self.prompt_angle_input()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.outline = []
        self.ridges = []
        self.valleys = []
        self.sub_roofs = []
        self.angle_data = {}
        self.step = "outline"
        self.log_window.delete(1.0, tk.END)
        self.log("ステップ1: 外形を描画してください (左クリックで頂点追加、ダブルクリックで閉じる)")


if __name__ == "__main__":
    root = tk.Tk()
    app = RoofDrawer(root)
    root.mainloop()
