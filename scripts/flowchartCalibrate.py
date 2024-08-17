from graphviz import Digraph

def create_calibration_flowchart(output_file):
    dot = Digraph(comment='Stereo Camera Calibration Process')
    dot.attr(rankdir='TB', size='8,8')

    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')

    # Add nodes
    dot.node('A', 'Start')
    dot.node('B', 'Load Left and Right Camera Images')
    dot.node('C', 'Find Chessboard Corners')
    dot.node('D', 'Prepare Object Points')
    dot.node('E', 'Calibrate Left Camera')
    dot.node('F', 'Calibrate Right Camera')
    dot.node('G', 'Perform Stereo Calibration')
    dot.node('H', 'Perform Stereo Rectification')
    dot.node('I', 'Calculate Reprojection Error')
    dot.node('J', 'Save Calibration Data')
    dot.node('K', 'End')

    # Add edges
    dot.edge('A', 'B')
    dot.edge('B', 'C')
    dot.edge('C', 'D')
    dot.edge('D', 'E')
    dot.edge('D', 'F')
    dot.edge('E', 'G')
    dot.edge('F', 'G')
    dot.edge('G', 'H')
    dot.edge('H', 'I')
    dot.edge('I', 'J')
    dot.edge('J', 'K')

    # Add decision node
    dot.attr('node', shape='diamond', style='filled', fillcolor='lightyellow')
    dot.node('Dec1', 'Corners\nDetected?')
    dot.edge('C', 'Dec1')
    dot.edge('Dec1', 'D', label='Yes')
    dot.edge('Dec1', 'B', label='No')

    # Render the graph
    dot.render(output_file, format='png', cleanup=True)
    print(f"Flowchart saved as {output_file}.png")

def main():
    output_file = '../visualizations/calibration_flowchart'
    create_calibration_flowchart(output_file)

if __name__ == "__main__":
    main()