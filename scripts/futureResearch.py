import matplotlib.pyplot as plt
import numpy as np

def create_future_research_visualization(output_file):
    # Define research areas and their importance/priority
    areas = ['Improved Distortion Models', 'Real-time Calibration', 'Multi-camera Systems', 
             'Deep Learning Integration', 'Adaptive Calibration', 'Extreme Conditions Testing']
    importance = [0.8, 0.9, 0.7, 0.85, 0.75, 0.6]

    # Create a polar plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Compute the angle for each area
    angles = np.linspace(0, 2*np.pi, len(areas), endpoint=False)

    # Close the plot by appending the first value to the end
    values = importance + [importance[0]]
    angles = np.concatenate((angles, [angles[0]]))

    # Plot the data
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.3)

    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(areas)

    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0.2, 1.1, 0.2))
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])

    # Add title
    plt.title('Suggested Areas for Future Research', y=1.1)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Future research visualization saved as {output_file}")

def main():
    output_file = '../visualizations/future_research_areas.png'
    create_future_research_visualization(output_file)

if __name__ == "__main__":
    main()