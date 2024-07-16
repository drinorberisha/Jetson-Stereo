import os

def run_script(script_name):
    os.system("python2.7 {}".format(script_name))

def main():
    print("Starting process...")
    run_script('scripts/lc.py')
    run_script('scripts/new.py')
    # dr
    # run_script('scripts/AdaptiveHistogramEqualization.py')
    # run_script('scripts/UndistortImages.py')
    # run_script('scripts/Rectification.py')
    # run_script('scripts/sgmb3Cuda.py')
    print("Process completed!")

if __name__ == "__main__":
    main()
