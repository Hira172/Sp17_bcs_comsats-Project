from yolo.utils import Load_Yolo_model
import sys, getopt
from core import system
from yolo.configs import YOLO_INPUT_SIZE




def main(argv):


    # video_path   = "./Videos/Hira Redmi.mp4" ## tilted
    # video_path   = "./Videos/Hira Remi.mp4"
    video_path = "./Videos/Muneeb iphone 6.mp4"
    # video_path   = "./Videos/Aqasha realme 5.mp4"
    # video_path   = "./Videos/Muneeb iphone 6(1).mp4"
    tm = False # flag for momentory time of collusion
    rt = True # real time flag

    options = "acr"
    long_options = [ "acceleration", "camera", "realtime"]

    try:
        arguments, values = getopt.getopt(argv, options, long_options)

        for currentArgument, currentValue in arguments:

            if currentArgument in ("-a", "--acceleration"):
                print("\n\n\nDiplaying Momentary Time to Contact without Acceleration")
                tm = True


            elif currentArgument in ("-c", "--camera"):
                print("\n\n\nUsing camera")
                video_path = False


            elif currentArgument in ("-r", "--realtime"):
                print("\n\n\nNot using real time")
                rt = False



    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))



    yolo = Load_Yolo_model()
    system(yolo, video_path, "detection.avi", input_size=YOLO_INPUT_SIZE, show=True, iou_threshold=0.1,rectangle_colors=(255,0,0),Track_only = ['car','truck','motorbike','person'], display_tm = tm, realTime = rt )





if __name__ == "__main__":
   main(sys.argv[1:])