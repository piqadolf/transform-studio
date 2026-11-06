from manim import *
import numpy as np

SAMPLES = 36
LEFT = np.array([-3.9, -0.65, 0.0])
RIGHT = np.array([2.55, -0.65, 0.0])


def circle_points(count, radius):
    angles = 2 * np.pi * np.arange(count) / count
    return [radius * np.array([np.cos(t), np.sin(t), 0.0]) for t in angles]


def polygon_samples(vertices, count):
    vertices = [np.array([*vertex, 0.0], dtype=float) for vertex in vertices]
    edges = []
    perimeter = 0.0
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        length = np.linalg.norm(end - start)
        edges.append((start, end, length))
        perimeter += length

    points = []
    edge_index = 0
    passed = 0.0
    for sample_index in range(count):
        distance = sample_index * perimeter / count
        while distance > passed + edges[edge_index][2] and edge_index < len(edges) - 1:
            passed += edges[edge_index][2]
            edge_index += 1
        start, end, length = edges[edge_index]
        ratio = (distance - passed) / length
        points.append(start + ratio * (end - start))
    return points


def unique_differences(points):
    differences = {}
    for first in points:
        for second in points:
            vector = second - first
            key = tuple(np.round(vector[:2], 6))
            differences[key] = vector
    return sorted(
        differences.values(),
        key=lambda vector: (np.linalg.norm(vector), np.arctan2(vector[1], vector[0])),
    )


class DifferenceSets(Scene):
    def construct(self):
        radius = 1.15
        circle = Circle(radius=radius, color=BLUE_B, stroke_width=4).move_to(LEFT)
        circle_target = Circle(
            radius=2 * radius,
            color=BLUE_B,
            stroke_width=4,
            fill_color=BLUE_D,
            fill_opacity=0.16,
        ).move_to(RIGHT)
        self.show_difference_set(
            "Окружность",
            circle,
            circle_points(SAMPLES, radius),
            circle_target,
            MathTex(r"D=\{v:\ \|v\|\leq 2R\}", color=BLUE_B),
            [(0, 9), (4, 20), (10, 28), (0, 18)],
        )

        side = 2.1
        half = side / 2
        square_vertices = [(-half, -half), (half, -half), (half, half), (-half, half)]
        square = Polygon(
            *[LEFT + np.array([*vertex, 0.0]) for vertex in square_vertices],
            color=GREEN_B,
            stroke_width=4,
        )
        square_target = Square(
            side_length=2 * side,
            color=GREEN_B,
            stroke_width=4,
            fill_color=GREEN_D,
            fill_opacity=0.16,
        ).move_to(RIGHT)
        self.show_difference_set(
            "Квадрат",
            square,
            polygon_samples(square_vertices, SAMPLES),
            square_target,
            MathTex(r"D=[-a,a]\times[-a,a]", color=GREEN_B),
            [(0, 9), (4, 20), (10, 28), (0, 18)],
        )

        triangle_side = 2.2
        triangle_radius = triangle_side / np.sqrt(3)
        triangle_vertices = [
            (
                triangle_radius * np.cos(np.pi / 2 + 2 * np.pi * index / 3),
                triangle_radius * np.sin(np.pi / 2 + 2 * np.pi * index / 3),
            )
            for index in range(3)
        ]
        triangle = Polygon(
            *[LEFT + np.array([*vertex, 0.0]) for vertex in triangle_vertices],
            color=ORANGE,
            stroke_width=4,
        )
        triangle_points = [np.array([*vertex, 0.0]) for vertex in triangle_vertices]
        hexagon_vertices = []
        for first in triangle_points:
            for second in triangle_points:
                vector = first - second
                if np.linalg.norm(vector) > 1e-6:
                    hexagon_vertices.append(vector)
        hexagon_vertices = sorted(
            hexagon_vertices,
            key=lambda vector: np.arctan2(vector[1], vector[0]),
        )
        hexagon = Polygon(
            *[RIGHT + vertex for vertex in hexagon_vertices],
            color=ORANGE,
            stroke_width=4,
            fill_color=RED_D,
            fill_opacity=0.16,
        )
        self.show_difference_set(
            "Равносторонний треугольник",
            triangle,
            polygon_samples(triangle_vertices, SAMPLES),
            hexagon,
            Text("Правильный шестиугольник", font_size=30, color=ORANGE),
            [(0, 12), (4, 21), (10, 29), (0, 24)],
        )

    def show_difference_set(self, name, contour, points, target, result, pairs):
        title = Text(name, font_size=34).to_edge(UP, buff=0.25)
        formula = MathTex(r"D=\{p_i-p_j\mid p_i,p_j\in C\}")
        formula.next_to(title, DOWN, buff=0.18)
        left_header = Text("Исходный контур", font_size=24).move_to([LEFT[0], 2.05, 0])
        right_header = Text("Разности, перенесённые в O", font_size=24).move_to(
            [RIGHT[0], 2.05, 0]
        )
        axes = Axes(
            x_range=[-3.1, 3.1, 1],
            y_range=[-3.1, 3.1, 1],
            x_length=6.2,
            y_length=6.2,
            tips=False,
            axis_config={"stroke_color": GREY, "stroke_width": 1.5, "include_ticks": False},
        ).move_to(RIGHT)
        separator = Line([-0.8, -3.35, 0], [-0.8, 2.3, 0], color=GREY_E)
        origin = Dot(RIGHT, radius=0.05, color=WHITE)
        origin_label = MathTex("O", font_size=28).next_to(RIGHT, DR, buff=0.08)
        contour_points = VGroup(
            *[Dot(LEFT + point, radius=0.045, color=GREY_A) for point in points]
        )

        self.play(
            Write(title),
            Write(formula),
            FadeIn(left_header),
            FadeIn(right_header),
            Create(axes),
            Create(separator),
            Create(contour),
            FadeIn(origin),
            Write(origin_label),
            run_time=1.2,
        )
        self.play(LaggedStart(*[FadeIn(dot) for dot in contour_points], lag_ratio=0.03), run_time=1)

        for first_index, second_index in pairs:
            first = points[first_index]
            second = points[second_index]
            vector = first - second
            first_dot = Dot(LEFT + first, radius=0.075, color=RED)
            second_dot = Dot(LEFT + second, radius=0.075, color=YELLOW)
            first_label = MathTex("p_i", font_size=28).next_to(first_dot, UP, buff=0.08)
            second_label = MathTex("p_j", font_size=28).next_to(second_dot, DOWN, buff=0.08)
            left_arrow = Arrow(
                LEFT + second,
                LEFT + first,
                buff=0,
                color=YELLOW,
                stroke_width=4,
                max_tip_length_to_length_ratio=0.16,
            )
            right_arrow = Arrow(
                RIGHT,
                RIGHT + vector,
                buff=0,
                color=YELLOW,
                stroke_width=4,
                max_tip_length_to_length_ratio=0.12,
            )
            reverse_arrow = Arrow(
                RIGHT,
                RIGHT - vector,
                buff=0,
                color=BLUE_C,
                stroke_width=4,
                max_tip_length_to_length_ratio=0.12,
            )
            positive_dot = Dot(RIGHT + vector, radius=0.06, color=YELLOW)
            negative_dot = Dot(RIGHT - vector, radius=0.06, color=BLUE_C)

            self.play(
                FadeIn(first_dot),
                FadeIn(second_dot),
                Write(first_label),
                Write(second_label),
                GrowArrow(left_arrow),
                run_time=0.45,
            )
            self.play(TransformFromCopy(left_arrow, right_arrow), run_time=0.55)
            self.play(FadeIn(positive_dot), run_time=0.12)
            self.play(GrowArrow(reverse_arrow), FadeIn(negative_dot), run_time=0.35)
            self.play(
                FadeOut(first_dot),
                FadeOut(second_dot),
                FadeOut(first_label),
                FadeOut(second_label),
                FadeOut(left_arrow),
                FadeOut(right_arrow),
                FadeOut(reverse_arrow),
                run_time=0.25,
            )

        difference_dots = VGroup(
            *[
                Dot(RIGHT + vector, radius=0.032, color=BLUE_C)
                for vector in unique_differences(points)
            ]
        )
        batch_size = 48
        for start in range(0, len(difference_dots), batch_size):
            self.play(FadeIn(difference_dots[start : start + batch_size]), run_time=0.12)

        result.to_edge(DOWN, buff=0.22)
        self.play(FadeIn(target), Write(result), run_time=0.9)
        self.wait(1.2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)
        self.clear()
