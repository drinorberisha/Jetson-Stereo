from graphviz import Digraph

def create_implications_diagram(output_file):
    dot = Digraph(comment='Research Implications for Related Fields')
    dot.attr(rankdir='TB', size='12,12')

    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')

    # Central research topic
    dot.node('A', 'Underwater Stereo\nCamera Calibration\non Jetson Nano')

    # Related fields and implications
    dot.node('B1', 'Marine Robotics\n- Improved navigation\n- Better object detection')
    dot.node('B2', 'Underwater Archaeology\n- Enhanced artifact mapping\n- Precise 3D reconstruction')
    dot.node('B3', 'Environmental Monitoring\n- Accurate reef health assessment\n- Fish population tracking')
    dot.node('B4', 'Offshore Industry\n- Safer underwater inspections\n- Efficient maintenance')
    dot.node('B5', 'Computer Vision\n- Advanced underwater image processing\n- New calibration techniques')

    # Connect central topic to related fields
    dot.edge('A', 'B1')
    dot.edge('A', 'B2')
    dot.edge('A', 'B3')
    dot.edge('A', 'B4')
    dot.edge('A', 'B5')

    # Render the graph
    dot.render(output_file, format='png', cleanup=True)
    print(f"Implications diagram saved as {output_file}.png")

def main():
    output_file = '../visualizations/research_implications_diagram'
    create_implications_diagram(output_file)

if __name__ == "__main__":
    main()