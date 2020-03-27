import utils.funciones as funciones
import cv2
import re
import os
import logging
from utils.person import Person
from utils.funciones import read_labels_txt
from pathlib import Path


logger = logging.getLogger(__name__)

class DataProcessor():

    def __init__(self, model_path, input_dim=(257,257), threshold = 0.5):
        
        self.model, self.input_details, self.output_details = funciones.load_model(model_path)
        self.input_dim = input_dim
        self.threshold = 0.5
        self.rescale = (1,1)


    
    def process_frame(self, image_path):
        """Receives a frame path and returns the person associated
        
        Args:
            image_path (str): String containig the path of an image
        
        Returns:
            Person: Person associated to that frame.
        """
        frame = cv2.imread(image_path)
        frame = funciones.prepare_frame(frame, self.input_dim)
        output_data, offset_data = funciones.get_model_output(self.model,
                                                            frame,
                                                            self.input_details,
                                                            self.output_details)
        return Person(output_data, offset_data, self.rescale, self.threshold)
    
    def get_frame_groups(self,labels_path, actions, n=5):
        """From a labels path, a list of actions and a number of frames per
        training data row gets all posible groups of frames to process.
        
        Args:
            labels_path (str): Path to the labels.txt file
            actions (list): Actions to process
            n (int, optional): Size of the list of people needed. Defaults to 5.
        
        Returns:
            [type]: [description]
        """
        frame_groups = {}
        self.people_dict = {}
        labels = read_labels_txt(labels_path, actions)
        for label in labels:
            #Groups of frames longer than n
            valid_frame_interval = [group for group in labels[label] if group[1]-group[0]>= n-1]
            lst = []
            for frame_interval in valid_frame_interval:
                lst += self.expanded_groups(frame_interval, n)
            frame_groups[label] = lst
        return frame_groups
    def frame_interval_to_people_list(self,file, interval):
        PATH = Path(__file__).parents[1].joinpath("resources/{}".format(file))
        
        return [self.process_frame(str(PATH)+"/{}_frame_{}.jpg".format(file, i))\
            for i in range(interval[0], interval[1]+1)]
            


        

    def expanded_groups(self,frame_interval, n):
        """[summary]
        
        Args:
            frame_interval (list): Iterval of valid frames
            n (int): Sizew of the frame group
        
        Returns:
            dict: for each video, the valid groups.
        """
        n_groups = (frame_interval[1]-frame_interval[0]+2)-n
        
        return [list(range(frame_interval[0]+i, frame_interval[0]+n+i)) for i in range(n_groups)]

    def process_action(self):
        pass
        

    @staticmethod
    def process_video(filename, output_shape = (256,256), fps_reduce = 2):
        """Process a video from the resources folder and saves all the frames
        inside a folder with the name of the video
        FILENAME_frame_X.jpg
        
        Args:
            filename (str): Name of the video inside resources
            output_shape (tuple, optional): Size of the output images. Defaults to (256,256).
            fps_reduce (int, optional): Take one image out of  #fps_reduce. 
            Defaults to 2.
        """
        PATH = './resources/{}/'.format(filename.split(".")[0])
        print(PATH)
        try:
            os.mkdir(PATH)
        except:
            os.system("rm -r ./resources/{}".format(filename.split(".")[0]))
            os.mkdir(PATH)

        #Read video
        video = cv2.VideoCapture("./resources/{}".format(filename))
        count=0
        logger.debug("Started reading frames.")
        while video.isOpened():
            logger.debug("Reading frame {}/{} from file {}".format(count+1, video.get(cv2.CAP_PROP_FRAME_COUNT),filename))
            
            #Frame reading, reshaping and saving
            ret,frame = video.read()
            frame = cv2.resize(frame, (256,256))
            if count % fps_reduce == 0:
                cv2.imwrite(PATH+"{}_frame_{}.jpg".format(filename.split(".")[0],count), frame)
            count = count + 1
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):
                # If the number of captured frames is equal to the total number of frames,
                break
        logger.debug("Stop reading files.")
        video.release()
