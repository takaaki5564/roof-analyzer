import tkinter as tk
from tkinter import messagebox
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union, polygonize
import random
import math


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
            if len(self.ridges) < 1:
                messagebox.showerror("エラー", "1つ以上の稜線が必要です")
                return
            self.log("稜線の端点を統合します...")
            self.merge_ridge_endpoints()
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
            self.log("ステップ6: 屋根を傾斜方向に展開します")
            self.expand_sub_roofs()
            self.step = "done"

        elif self.step == "done":
            self.log("全てのステップが完了しました。再度外形を描画するか、終了してください")
            self.step = "outline"
            self.clear_canvas()

    def merge_ridge_endpoints(self, threshold=20):

        def find_nearest_point(target, points, threshold=20):
            nearest = None
            min_dist = float('inf')
            for p in points:
                dist = Point(target).distance(Point(p))
                if dist < min_dist and dist <= threshold:
                    min_dist = dist
                    nearest = p
            return nearest

        endpoints = set()
        for ridge in self.ridges:
            endpoints.add(ridge[0][0])  # start point
            endpoints.add(ridge[0][-1])  # end point

        merged_points = {}
        new_ridges = []

        for pt in endpoints:
            nearest = find_nearest_point(pt, merged_points.keys(), threshold)
            if nearest:
                merged_points[pt] = merged_points[nearest]
            else:
                merged_points[pt] = pt

        # Get ridges after merging
        for ridge in self.ridges:
            new_line = [merged_points[pt] for pt in ridge[0]]
            new_ridges.append((new_line, LineString(new_line)))

        self.ridges = new_ridges

        # new_endpoints = {}

        # # Merge endpoints of ridge to the nearest edge line
        # for pt in endpoints:
        #     point = Point(pt)
        #     min_dist = float('inf')
        #     nearest_point = pt

        #     for ridge in self.ridges:
        #         ridge_line = LineString(ridge[0])
        #         projected_dist = ridge_line.project(point)
        #         projected_point = ridge_line.interpolate(projected_dist)

        #         dist = point.distance(projected_point)

        #         if dist < min_dist and dist <= threshold:
        #             min_dist = dist
        #             nearest_point = projected_point

        #     new_endpoints[pt] = nearest_point

        # new_ridges = []
        # for ridge in self.ridges:
        #     new_line = [new_endpoints.get(pt, pt) for pt in ridge[0]]
        #     new_ridges.append((new_line, LineString(new_line)))

        # self.ridges = new_ridges

        # Draw merged ridges
        self.canvas.delete("all")

        # Draw outline
        self.canvas.create_polygon(self.outline, fill="white", outline="black", width=2)
        for pt in self.outline:
            self.canvas.create_oval(pt[0]-2, pt[1]-2, pt[0]+2, pt[1]+2, fill='black')

        for ridge in self.ridges:
            print(f"ridge: {ridge[0]}")

            self.canvas.create_line(ridge[0], fill="red", width=2)
            for pt in ridge[0]:
                self.canvas.create_oval(pt[0]-2, pt[1]-2, pt[0]+2, pt[1]+2, fill='red')

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
        split_polygons = [
            poly for poly in polygons
            if poly.is_valid and poly.within(main_polygon) and not poly.equals(main_polygon) and poly.area > 0]

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

    def update_polygon_outlines(self, color="black"):
        for poly in self.sub_roofs:
            self.draw_polygon_outline(poly, outline_color=color)

    def draw_polygon_outline(self, polygon, outline_color="black", width=2):
        points = list(polygon.exterior.coords)
        self.canvas.create_polygon(points, fill="", outline=outline_color, width=width)

    def draw_ridges(self):
        for ridge in self.ridges:
            self.canvas.create_line([coord for pt in ridge[0] for coord in pt], fill="red", width=2)

    def draw_slope_arrows(self):
        """ Calculate maximum tilt angle for each sub-roof and draw arrows """
        for poly in self.sub_roofs:
            centroid = poly.centroid

            # Find nearest edge
            nearest_ridge_point = None
            min_dist = float('inf')

            for ridge in self.ridges:
                ridge_line = LineString(ridge[0])
                projected_dist = ridge_line.project(centroid)
                projected_point = ridge_line.interpolate(projected_dist)

                dist = centroid.distance(projected_point)
                if dist < min_dist:
                    min_dist = dist
                    nearest_ridge_point = (projected_point.x, projected_point.y)

            if not nearest_ridge_point:
                continue

            dx = nearest_ridge_point[0] - centroid.x
            dy = nearest_ridge_point[1] - centroid.y

            arrow_dx = -dx
            arrow_dy = -dy

            arrow_len = 80
            norm = math.sqrt(arrow_dx**2 + arrow_dy**2)
            if norm == 0:
                continue

            arrow_dx = (arrow_dx / norm) * arrow_len
            arrow_dy = (arrow_dy / norm) * arrow_len

            end_x = centroid.x + arrow_dx
            end_y = centroid.y + arrow_dy

            self.canvas.create_line(centroid.x, centroid.y, end_x, end_y, arrow=tk.LAST, fill="blue", width=2)

    def prompt_angle_input(self):
        """ Prompt the user to input the angle of the current sub roof """
        if self.current_roof >= len(self.sub_roofs):
            self.log("全ての部分屋根の角度を入力しました。終了します")
            self.clear_angle_widgets()
            return

        # Draw all roofs with black color lines
        self.update_polygon_outlines(color="black")

        # Draw current roof with red color lines
        current_poly = self.sub_roofs[self.current_roof]
        self.draw_polygon_outline(current_poly, outline_color="red", width=3)

        # Input angle by the user
        self.log(f"{self.current_roof+1}番目の部分屋根の角度を入力してください")

        angle_var = tk.StringVar()
        self.angle_data[self.current_roof] = angle_var

        entry = tk.Entry(self.root, textvariable=angle_var)
        entry.pack()
        button = tk.Button(self.root, text="次へ", command=lambda: self.on_angle_input(angle_var))
        button.pack()

        if not hasattr(self, "angle_widgets"):
            self.angle_widgets = []

        self.angle_widgets.append(entry)
        self.angle_widgets.append(button)

    def clear_angle_widgets(self):
        if hasattr(self, "angle_widgets") and self.angle_widgets:
            print("Clearing angle widgets")
            for widget in self.angle_widgets:
                if widget.winfo_exists():
                    widget.destroy()
            self.angle_widgets.clear()

    def on_angle_input(self, angle_var):
        """ Process the angle input and proceed to the next sub roof """
        angle = angle_var.get()
        if not angle.isdigit():
            messagebox.showerror("エラー", "角度を入力してください")
            return

        print("Angle:", angle)
        # Get angle and draw in the center of sub-roof
        angle = int(angle)
        current_poly = self.sub_roofs[self.current_roof]
        centroid = current_poly.centroid

        # Draw angle text
        self.canvas.create_text(
            centroid.x, centroid.y, text=f"{angle}°", font=("Arial", 12, "bold"), fill="blue"
        )

        self.update_polygon_outlines(color="black")

        self.draw_ridges()

        self.current_roof += 1
        if self.current_roof >= len(self.sub_roofs):
            self.log("全ての部分屋根の角度を入力しました。傾斜方向を描画します")
            self.clear_angle_widgets()
            self.draw_slope_arrows()
            return

        self.prompt_angle_input()

    def get_max_tilt_vector(self, poly, scale_factor):

        centroid = poly.centroid

        nearest_ridge_points = None
        min_dist = float('inf')

        for ridge in self.ridges:
            ridge_line = LineString(ridge[0])
            projected_dist = ridge_line.project(centroid)
            projected_point = ridge_line.interpolate(projected_dist)

            dist = centroid.distance(projected_point)
            if dist < min_dist:
                min_dist = dist
                nearest_ridge_points = (projected_point.x, projected_point.y)
        if not nearest_ridge_points:
            return None, None, None, None

        dx = nearest_ridge_points[0] - centroid.x
        dy = nearest_ridge_points[1] - centroid.y

        return dx * scale_factor, dy * scale_factor, nearest_ridge_points[0], nearest_ridge_points[1]

    def draw_transformed_roofs(self):
        self.canvas.delete("all")

        # Draw expanded sub roofs
        for poly in self.sub_roofs:
            r, g, b = [random.randint(180, 255) for _ in range(3)]
            color = f"#{r:02x}{g:02x}{b:02x}"

            self.canvas.create_polygon(
                [coord for pt in poly.exterior.coords for coord in pt],
                fill=color, outline="black", width=2
            )

        # Draw original sub roofs
        for poly in self.original_sub_roofs:
            self.canvas.create_polygon(
                [coord for pt in poly.exterior.coords for coord in pt],
                fill="", outline="gray", width=1, dash=(3, 3)
            )

        self.draw_slope_arrows()

    def expand_sub_roofs(self):
        """ Expand sub roofs to the direction of max tilt angle with 1/cos(a) """
        self.original_sub_roofs = list(self.sub_roofs)

        new_sub_roofs = []

        for i, poly in enumerate(self.sub_roofs):
            if i not in self.angle_data:
                continue

            angle_str = self.angle_data[i].get().strip()
            if not angle_str.isdigit():
                self.log(f"{i+1}番目の部分屋根の角度が不正です")
                continue

            angle_deg = int(angle_str)
            angle_rad = math.radians(angle_deg)
            scale_factor = 1 / math.cos(angle_rad)

            # Get the maximum tilt vector
            # TODO: tilt vector should be input from customer
            dx, dy, nx, ny = self.get_max_tilt_vector(poly, scale_factor)

            if dx is None:
                continue
            norm = math.sqrt(dx**2 + dy**2)
            if norm == 0:
                continue

            slope_vector = (dx / norm, dy / norm)  # normalized slope vector
            perpendicular_vecor = (-slope_vector[1], slope_vector[0])  # Unit vecor perpendicular to slope_vector

            # Transform the polygon
            transformed_coords = []
            for x, y, in poly.exterior.coords:
                # Expand to the direction of the slope vector
                dx_rel = x - nx  # TODO: Fix point should be input from customer
                dy_rel = y - ny

                slop_component = dx_rel * slope_vector[0] + dy_rel * slope_vector[1]
                perp_component = dx_rel * perpendicular_vecor[0] + dy_rel * perpendicular_vecor[1]

                slop_component *= scale_factor

                new_x = nx + slop_component * slope_vector[0] + perp_component * perpendicular_vecor[0]
                new_y = ny + slop_component * slope_vector[1] + perp_component * perpendicular_vecor[1]

                transformed_coords.append((new_x, new_y))

            new_sub_roofs.append(Polygon(transformed_coords))

        self.sub_roofs = new_sub_roofs

        # Clear canvas and draw the expanded sub roofs
        self.draw_transformed_roofs()

        self.log("屋根の展開が完了しました")

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
