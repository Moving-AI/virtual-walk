import utils.funciones as funciones
import cv2
import re
import os
import logging
from utils.person import Person
from utils.funciones import read_labels_txt
from pathlib import Path
from itertools import chain 


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
    
    def get_frame_groups(self, actions,labels_path = None, n=5):
        """From a labels path, a list of actions and a number of frames per
        training data row gets all posible groups of frames to process.
        
        Args:
            labels_path (str): Path to the labels.txt file
            actions (list): Actions to process
            n (int, optional): Size of the list of people needed. Defaults to 5.
        
        Returns:
            [type]: [description]
        """
        if labels_path is None:
            labels_path = Path(__file__).parents[1].joinpath("resources/{}".format("labels.txt"))
        frame_groups = {}
        self.people_dict = {}
        labels = read_labels_txt(labels_path, actions)
        for label in labels:
            #Groups of frames longer than n
            valid_frame_intervals = [group for group in labels[label] if group[1]-group[0]>= n-1]
            #Transform each interval in a list of valid persons
            frame_person_list = [self.get_valid_persons(label, interval, n) for interval in valid_frame_intervals]
            #Get groups of n contiguous persons
            valid_persons_groups = [self.valid_groups(lst, n) for lst in frame_person_list]
            filter_nones = [element for element in valid_persons_groups if element is not None]
            #Complete dictionary
            frame_groups[label] = filter_nones

            
        return frame_groups
    
    def get_valid_persons(self,file, interval, n):
        persons_list = self.frame_interval_to_people_list(file, interval)
        persons_list = [element for element in persons_list if element[1].H > 0]
         

    def frame_interval_to_people_list(self,file, interval):
        """From an interval [start, end] of frames from video, returns a list
        of tuples (index, person(i_Frame)).
        
        Args:
            file (str): folder containing frames
            interval (list): start and end of the interval
        
        Returns:
            list: List of Persons calculated from images
        """
        PATH = Path(__file__).parents[1].joinpath("resources/{}".format(file))
        
        return [[i,self.process_frame(str(PATH)+"/{}_frame_{}.jpg".format(file, i))]\
            for i in range(interval[0], interval[1]+1)]
            
    def valid_groups(self, lst, n):
        """Given a list of persons, returns the valid lists of contiguous persons
        (frames)
        
        Args:
            n (int): Size of the desired lists of persons
            lst (list): List of lists [int, Person]
        """
        valid, result, aux = 0 , [], []
        if lst is not None:
            for index, i in enumerate(lst):
                if valid == 0:
                    #New group
                    aux.append(i)
                    valid += 1
                elif i[0] - aux[valid-1][0] == 1 and valid < n-1:
                    #Value is valid
                    aux.append(i)
                    valid += 1
                elif i[0] - aux[valid-1][0] == 1 and valid == n-1:
                    #Group is now complete
                    aux.append(i)
                    result.append(aux) 
                    aux = []
                    valid = 0
                else:
                    aux = []
                    valid = 0
            return result
        else: return None

        

    #def expanded_groups(self,frame_interval, n):
    #    """[summary]
    #    
    #    Args:
    #        frame_interval (list): Iterval of valid frames
    #        n (int): Sizew of the frame group
    #    
    #    Returns:
    #        dict: for each video, the valid groups.
    #    """
    #    n_groups = (frame_interval[1]-frame_interval[0]+2)-n
    #    
    #    return [list(range(frame_interval[0]+i, frame_interval[0]+n+i)) for i in range(n_groups)]

    #def process_action(self):
    #    pass
        

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
        #print(PATH)
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
                cv2.imwrite(PATH+"{}_frame_{}.jpg".format(filename.split(".")[0],count//fps_reduce), frame)
            count = count + 1
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):
                # If the number of captured frames is equal to the total number of frames,
                break
        logger.debug("Stop reading files.")
        video.release()
