from utils.set_up import set_up
import sys

def main():
    logger, project_root = set_up()
    sys.path.append(project_root)
    
    logger.info('Finished running the code')

    
if __name__ == "__main__":
    main()  # Call the main function if this script is executed as the main program