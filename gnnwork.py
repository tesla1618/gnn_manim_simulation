from manim import *

config.ffmpeg_executable = r"C:\Users\Admin\Downloads\ffmpeg-2024-06-06-git-d55f5cba7b-full_build\ffmpeg-2024-06-06-git-d55f5cba7b-full_build\bin\ffmpeg.exe"
class ParticleCollisionToGNN(Scene):
    def construct(self):
        # Step 1: Particle Collision
        self.display_step(
            "Step 1: Particle Collision", 
            "Simulating the collision of charged and neutral particles."
        )
        
        particles = VGroup(
            Circle(radius=0.2, color=RED).move_to(LEFT * 3),
            Circle(radius=0.2, color=BLUE).move_to(RIGHT * 3)
        )
        self.play(Create(particles))
        self.wait(1)

       # Simulate collision
        self.play(
            particles[0].animate.move_to(ORIGIN),
            particles[1].animate.move_to(ORIGIN),
            run_time=2
        )
        
        collision_effect = VGroup(
            Circle(radius=1, color=YELLOW, fill_opacity=0.5).scale(2),
            Circle(radius=1, color=YELLOW, fill_opacity=0.5).scale(1.5)
        )
        self.play(Create(collision_effect))
        self.wait(1)
        
        # Add label with arrow
        label = Tex("Collision Effect", font_size=24).move_to(UP * 2.5)
        arrow = Arrow(start=label.get_bottom(), end=collision_effect[0].get_top(), buff=0.1)

        self.play(Write(label), Create(arrow))
        self.wait(1)

        self.play(FadeOut(collision_effect), FadeOut(particles), FadeOut(label), FadeOut(arrow))


        # Step 2: Graph Construction
        self.display_step(
            "Step 2: Graph Construction", 
            "Constructing a graph from the particles where nodes represent particles and edges represent interactions."
        )

        # Nodes and Edges
        node_config = {"radius": 0.25, "color": BLUE}
        node_A = Circle(**node_config).move_to(LEFT * 3 + DOWN)
        node_B = Circle(**node_config).move_to(LEFT + UP * 2)
        node_C = Circle(**node_config).move_to(RIGHT * 3 + DOWN)
        node_D = Circle(**node_config).move_to(RIGHT * 2 + UP * 2)
        node_E = Circle(**node_config).move_to(ORIGIN)

        nodes = VGroup(node_A, node_B, node_C, node_D, node_E)
        node_labels = VGroup(
            Tex("A").next_to(node_A, DOWN),
            Tex("B").next_to(node_B, UP),
            Tex("C").next_to(node_C, DOWN),
            Tex("D").next_to(node_D, UP),
            Tex("E").next_to(node_E, DOWN)
        )

        edges = VGroup(
            Line(node_A.get_center(), node_B.get_center()),
            Line(node_A.get_center(), node_E.get_center()),
            Line(node_E.get_center(), node_B.get_center()),
            Line(node_E.get_center(), node_C.get_center()),
            Line(node_E.get_center(), node_D.get_center()),
            Line(node_C.get_center(), node_D.get_center())
        )

        self.play(Create(nodes), Create(node_labels), Create(edges))
        self.wait(2)

        self.play(FadeOut(nodes), FadeOut(node_labels), FadeOut(edges))

        # Step 3: Random Selection and Feature Masking
        self.display_step(
            "Step 3: Random Selection and Feature Masking", 
            "Randomly selecting a portion of charged particles and replacing their features with those of neutral particles to prevent overfitting."
        )

        # Nodes with Labels
        charged_LV = Circle(radius=0.25, color=GREEN).move_to(LEFT * 3 + DOWN)
        charged_PU = Circle(radius=0.25, color=RED).move_to(LEFT + UP * 2)
        neutral_LV = Circle(radius=0.25, color=BLUE).move_to(RIGHT * 3 + DOWN)
        neutral_PU = Circle(radius=0.25, color=PURPLE).move_to(RIGHT * 2 + UP * 2)
        masked = Circle(radius=0.25, color=ORANGE).move_to(ORIGIN)

        nodes_b = VGroup(charged_LV, charged_PU, neutral_LV, neutral_PU, masked)
        node_labels_b = VGroup(
            Tex("Charged LV").next_to(charged_LV, DOWN),
            Tex("Charged PU").next_to(charged_PU, UP),
            Tex("Neutral LV").next_to(neutral_LV, DOWN),
            Tex("Neutral PU").next_to(neutral_PU, UP),
            Tex("Masked").next_to(masked, DOWN)
        )

        self.play(Create(nodes_b), Create(node_labels_b))
        self.wait(2)

        masking_text = Tex("Feature Masking", font_size=36).next_to(masked, RIGHT)
        masking_explanation = Tex(
            "Randomly select 10% charged particles and replace their features with neutral-specific features",
            font_size=24
        ).next_to(masking_text, DOWN)
        self.play(Write(masking_text), Write(masking_explanation))
        self.wait(3)

        self.play(FadeOut(nodes_b), FadeOut(node_labels_b), FadeOut(masking_text), FadeOut(masking_explanation))

        # Step 4: GNN Encoding
        title_text = Tex("Step 4: GNN Encoding", font_size=40).to_edge(UP)
        description_text = Tex(
            "Encoding the graph using a Graph Neural Network (GNN).", 
            font_size=24
        ).next_to(title_text, DOWN)
        self.play(Write(title_text), Write(description_text))
        self.wait(2)

        # Add GNN Layers
        input_layer = VGroup(
            Circle(radius=0.15, color=WHITE),
            # Tex("Input").next_to(UP * 0.5, DOWN).shift(1.2 * RIGHT)
        )
        hidden_layers = VGroup(
            VGroup(
                Circle(radius=0.15, color=WHITE),
                # Tex("Hidden 1").next_to(UP * 0.5, DOWN).shift(1.2 * RIGHT)
            ),
            VGroup(
                Circle(radius=0.15, color=WHITE),
                # Tex("Hidden 2").next_to(UP * 0.5, DOWN).shift(1.2 * RIGHT)
            )
        )
        output_layer = VGroup(
            Circle(radius=0.15, color=WHITE),
            # Tex("Output").next_to(UP * 0.5, DOWN).shift(1.2 * RIGHT)
        )
        gnn_layers = VGroup(input_layer, *hidden_layers, output_layer)
        gnn_layers.arrange(DOWN, buff=0.5)
        gnn_layers.move_to(ORIGIN)

        self.play(Create(gnn_layers))

        # Animate flow of information through layers
        prev_layer = None
        for layer in hidden_layers:
            if prev_layer:
                edge = Arrow(prev_layer, layer, buff=0)
                self.play(Create(edge))
                self.play(
                    FadeIn(layer),
                    FadeOut(edge)
                )
            else:
                self.play(FadeIn(layer))
            prev_layer = layer

        self.wait(2)

        # Add details about GNN operations
        encoding_text = Tex(
            "Graph nodes are updated through GNN layers using message passing and gating mechanisms.",
            font_size=20
        ).to_edge(DOWN, buff=1.5)
        self.play(Write(encoding_text))
        self.wait(3)

        # Fade out entire step 4
        self.play(FadeOut(title_text), FadeOut(description_text), FadeOut(gnn_layers), FadeOut(encoding_text))
        self.wait(1)

        # Step 5: Prediction
        self.display_step(
            "Step 5: Prediction", 
            "Making final predictions using the encoded graph information."
        )

        final_prediction = Tex("Final Prediction", font_size=36).move_to(ORIGIN)
        prediction_explanation = Tex(
            "Using GNN encoding, predict particle labels based on their features and relationships",
            font_size=24
        ).next_to(final_prediction, DOWN)
        self.play(Write(final_prediction), Write(prediction_explanation))
        self.wait(3)

    def display_step(self, title, description):
        title_text = Tex(title, font_size=40).to_edge(UP)
        description_text = Tex(description, font_size=24).next_to(title_text, DOWN)
        self.play(Write(title_text), Write(description_text))
        self.wait(2)
        self.play(FadeOut(title_text), FadeOut(description_text))

if __name__ == "__main__":
    config.pixel_height = 1080
    config.pixel_width = 1920
    config.frame_rate = 60

    import subprocess
    subprocess.run(["manim", "-pqh", "gnnwork.py"])