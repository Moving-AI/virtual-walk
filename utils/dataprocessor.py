
import cv2, re, os, logging, imutils, json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import chain
from utils.person import Person
from utils.funciones import read_labels_txt
import utils.funciones as funciones
from utils.person_frames import PersonMovement

FORMAT = "%(asctime)s - %(levelname)s: %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)

formatter = logging.Formatter(FORMAT)
logger.setLevel(logging.INFO)

class DataProcessor():

    def __init__(self, model_path, input_dim=(257,257), threshold = 0.5):
        
        self.model, self.input_details, self.output_details = funciones.load_model(model_path)
        self.input_dim = input_dim
        self.threshold = 0.5
        self.rescale = (1,1)

    def training_file_writer(self, labels_path = None, output_file = None, append = False, n = 5, times_v = 10):
        """This function is the main function inside DataProcessor file. It runs the whole pipeline, in this order:
        - Gets actions and frame intervals from the labels file
        - Processes the frame intervals, keeping only the valid ones.
        - Groups the frames in groups of n
        - Coordinates are calculated from those groups
        - The coordinates are added to the output file in .csv format
        
        Args:
            labels_path (str, optional): Absolute path of the labels file. If none is taken from
            action-detection/resources.
            output_file (str, optional): Absolute path of the output csv file. If none is saved into 
            action-detection/resources/training_data.csv.
            append (bool, optional): If True, the calculated coordinates are ADDED to the file
            if it's not empty. Defaults to False.
            n (int, optional): Number of frames to obtain coordinates from. Defaults to 5.
            times_v (int, optional): Times point speed is introduced into coordinates. Defaults to 10.
        
        Returns:
            pandas.DataFrame: DataFrame containing the obtained coordinates and the ones in output_file
            if append = True
        """
        if labels_path is None:
            labels_path = Path(__file__).parents[1].joinpath("resources/{}".format("labels.txt"))
        if output_file is None:
            output_file = Path(__file__).parents[1].joinpath("resources/{}".format("training_data.csv"))
        
        #Obtain the dictionary of coordinates
        coordinates_dict = self.get_coordinates(labels_path, n = n, times_v = times_v)
        try:
            if append:
                df_initial = pd.read_csv(output_file)
                df_list = [df_initial]
            else:
                df_list = []
        except:
            if append:
                print("_________________________")
                logger.warning("Append is set to true but the reading gave an exception")
            df_list = []

        for video in coordinates_dict:
            if len(coordinates_dict[video]) == 0:
                continue
            else:
                array = np.vstack(coordinates_dict[video])
                df = pd.DataFrame(array)
                action = video.split("_")[0]
                df["action"] = [action]*len(coordinates_dict[video])
                df_list.append(df)
        cols_model_orig = [int(x) for x in list(df_list[-1].columns) if str(x).isnumeric()]
        cols_model_target = [str(x) for x in cols_model_orig if str(x).isnumeric()]
        mapper = {}
        for orig, target in zip(cols_model_orig, cols_model_target):
            mapper[orig] = target
        
        df_list = [df_iter.rename(mapper, axis = 'columns') for df_iter in df_list]
        
        logger.info("Concatenating {} DataFrames before writing.".format(len(df_list)))
        
        df = pd.concat(df_list, axis = 0, ignore_index = True)

        df.to_csv(output_file, index=False)
        return df


    def get_coordinates(self, labels_path=None, actions=None, n = 5, times_v = 10):
        """This functions is a wrapper that makes this steps:
        - Gets actions and frame intervals from the labels file
        - Processes the frame intervals, keeping only the valid ones.
        - Groups the frames in groups of n
        - Coordinates are calculated from those groups
        Args:
            labels_path (str, optional): Absolute for the labels file. If none, it is searched inside
            action-recognition/resources
            actions (list, optional): Actions contained in the label file. If None, the program searches
            them
            n (int, optional): Lenght of the frame list to process. Defaults to 5.
            times_v (int, optional): Times speeds of the points is introduced as coordinate. Defaults to 10.
        
        Returns:
            dict: Dictionary that contains for each video in the labels file the coordinates after running the
            frame selection pipeline.
        """

        logger.info("Calculating coordinates from labels_path {}".format(labels_path))
        if labels_path is None:
            labels_path = Path(__file__).parents[1].joinpath("resources/{}".format("labels.txt"))
        else:
            labels_path = Path(labels_path)
        actions = DataProcessor.find_actions(labels_path)
        frame_groups = self.get_frame_groups(actions, labels_path, n)
        coordinates_dict = {}
        #print(frame_groups)
        for video in frame_groups:
            logger.debug("Calculating coordinates for video {}".format(video))
            for group in frame_groups[video]:
                if len(group) == 0:
                    continue
                else:
                    
                    if video not in coordinates_dict: coordinates_dict[video] = []
                    persons = [element[1] for element in group]
                    coordinates = PersonMovement(persons, times_v).coords
                    coordinates_dict[video].append(coordinates)
        return coordinates_dict



    
    def process_frame(self, image_path):
        """Receives a frame path and returns the person associated
        
        Args:
            image_path (str): String containig the path of an image
        
        Returns:
            Person: Person associated to that frame.
        """
        logger.debug("Processing frame {}".format(image_path.split("/")[-1]))
        frame = cv2.imread(image_path)
        frame = funciones.prepare_frame(frame, self.input_dim)
        output_data, offset_data = funciones.get_model_output(self.model,
                                                            frame,
                                                            self.input_details,
                                                            self.output_details)
        return Person(output_data, offset_data, self.rescale, self.threshold)
    
    def get_frame_groups(self, actions,labels_path, n=5):
        """From a labels path, a list of actions and a number of frames per
        training data row gets all posible groups of frames to process.

        Takes into consideration 
        
        Args:
            labels_path (str): Path to the labels.txt file
            actions (list): Actions to process
            n (int, optional): Size of the list of people needed. Defaults to 5.
        
        Returns:
            [type]: [description]
        """
        logger.info("Getting frame groups for labels in {}".format(labels_path))
        frame_groups = {}
        self.people_dict = {}
        labels = read_labels_txt(str(labels_path), actions)
        for label in labels:
            logger.debug("Getting grame groups for label {}".format(label))
            #Groups of frames longer than n
            valid_frame_intervals = [group for group in labels[label] if group[1]-group[0]>= n-1]
            #Transform each interval in a list of valid persons
            frame_person_list = [self.get_valid_persons(label, interval, n) for interval in valid_frame_intervals]
            #Get groups of n contiguous persons
            valid_persons_groups = [self.valid_groups(lst, n) for lst in frame_person_list]
            filter_nones = [element for element in valid_persons_groups if element is not None]
            #Complete dictionary
            frame_groups[label] = filter_nones
        #There is an undesired extra level in the lists generated. We remove it
        frame_groups_definitive = {}
        logging.info("Cleaning frame groups.")
        for video in frame_groups:
            frame_groups_definitive[video] = list(chain.from_iterable(frame_groups[video]))    
        return frame_groups_definitive
    
    def get_valid_persons(self,fle, interval, n):
        logger.debug("Getting valid persons from file {}, interval {}".format(fle, interval))
        persons_list = self.frame_interval_to_people_list(fle, interval)
        persons_list = [element for element in persons_list if element[1].is_valid()]
        return persons_list
         

    def frame_interval_to_people_list(self,fle, interval):
        """From an interval [start, end] of frames from video, returns a list
        of tuples (index, person(i_Frame)).
        
        Args:
            file (str): folder containing frames
            interval (list): start and end of the interval
        
        Returns:
            list: List of Persons calculated from images
        """
        logger.debug("Calculating people list from interval {} in file {}".format(interval, fle))
        PATH = Path(__file__).parents[1].joinpath("resources/{}".format(fle))
        
        return [[i,self.process_frame(str(PATH)+"/{}_frame_{}.jpg".format(fle, i))]\
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
            for i in lst:
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
    def find_actions(file):
        actions = set()
        regex = r"[a-z]+"
        for line in open(str(file)):
            for match in re.finditer(regex, line):
                #print('Found on line {}: {}'.format(i+1, match.group()))
                actions.add(match.group())
        return list(actions)


    @staticmethod
    def process_video(filename, output_shape = (257,257), fps_reduce = 2, angle = 0):
        """Process a video from the resources folder and saves all the frames
        inside a folder with the name of the video
        FILENAME_frame_X.jpg
        
        Args:
            filename (str): Name of the video inside resources
            output_shape (tuple, optional): Size of the output images. Defaults to (256,256).
            fps_reduce (int, optional): Take one image out of  #fps_reduce. 
            Defaults to 2.
            angle (int): Angle that the video images should be rotated. 
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
            _,frame = video.read()
            frame = cv2.resize(frame, output_shape)
            #if DataProcessor.check_rotation("./resources/{}".format(filename)) is not None:

            frame = imutils.rotate(frame, angle)
            
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
    #@staticmethod
    #def check_rotation(path_video_file):
    ## this returns meta-data of the video file in form of a dictionary
    #    meta_dict = ffmpeg.probe(path_video_file)
#
    #    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    #    # we are looking for
    #    rotateCode = None
    #    if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
    #        rotateCode = cv2.ROTATE_90_CLOCKWISE
    #    elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
    #        rotateCode = cv2.ROTATE_180
    #    elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
    #        rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
#
    #    return rotateCode
