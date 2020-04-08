import logging
import math
import re
import time

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

FORMAT = "%(asctime)s - %(levelname)s: %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)

formatter = logging.Formatter(FORMAT)
logger.setLevel(logging.INFO)


class Controller:
    """Class to control the movement in Google Street View. It only works with Firefox.
    If there is no guess about where to start (initial_url or coordinates) it starts
    in Zaragoza (Spain).
    
    Returns:
        Controller:
    """
    def __init__(self, classes, initial_url=None, driver_path=None, time_rotation=0.5, coordinates=None):
        """Constructor for Controller class.
        
        Args:
            classes (list): Generally, ['walk', 'stand', 'left', 'right'], 
            but the program understands any order
            initial_url (str, optional): Position where the program is going to start.
            Defaults to None.
            driver_path (str, optional): The driver to where geckodriver is located.
            Defaults to None.
            time_rotation (float, optional): The time associated to a single rotation.
            Defaults to 0.5.
            coordinates (tuple, optional): Coordinates to begin the walk. Defaults to None.
        """
        if driver_path is None:
            self._driver = webdriver.Firefox()
        else:
            self._driver = webdriver.Firefox(executable_path=driver_path)

        self._driver.set_window_position(x=-10, y=0)

        if coordinates is not None:
            URL = "https://www.google.com/maps/@?api=1&map_action=pano" + "&viewpoint={:.7f},{:.7f}".format(
                coordinates[0], coordinates[1])
            self.distance_calculator = DistanceMeter(starting_coords=coordinates)
        elif initial_url is not None:
            URL = initial_url
            self.distance_calculator = DistanceMeter(starting_url=URL)
        else:
            URL = "https://www.google.es/maps/@41.6425054,-0.8932757,3a,86.3y,35.92h,83.74t/data=!3m6!1e1!3m4!1sB4DQl3bfNd-txTOR2bEjPg!2e0!7i16384!8i8192"
            self.distance_calculator = DistanceMeter(starting_url=URL)

        self._driver.get(URL)
        time.sleep(5)
        self.prepare_maps()
        self.classes = classes
        self.time_rotation = time_rotation
        self.action_mapper = self.map_actions()


    def prepare_maps(self):
        self._driver.find_element_by_tag_name("canvas").click()

    def rotate_left(self):
        endtime = time.time() + self.time_rotation
        while True:
            ActionChains(self._driver).key_down('a').perform()
            if time.time() > endtime:
                ActionChains(self._driver).key_up('a').perform()
                break

    def rotate_right(self):
        endtime = time.time() + self.time_rotation
        while True:
            ActionChains(self._driver).key_down('d').perform()
            if time.time() > endtime:
                ActionChains(self._driver).key_up('d').perform()
                break

    def step_forward(self):
        ActionChains(self._driver).send_keys(Keys.ARROW_UP).perform()

    def release_keys(self):
        ActionChains(self._driver).key_up(Keys.ARROW_UP).perform()
        ActionChains(self._driver).key_up('a').perform()
        ActionChains(self._driver).key_up('d').perform()

    def void(self):
        pass

    def map_actions(self):
        return {'walk': self.step_forward, 'left': self.rotate_left, 'right': self.rotate_right, 'stand': self.void}

    def perform_action(self, i_action):
        """Perform an action given its index within self.classes
        
        Args:
            i_action (int): index of classes list
        """

        action = self.classes[i_action]
        self.action_mapper[action]()
        self.distance_calculator.update_distance(self._driver.current_url)
        logging.info("You have made {} meters".format(self.distance_calculator.distance))

    def perform_action_name(self, action):
        """Perform action given its name.
        
        Args:
            action (str): Name of the action to perform. As it uses a mapper, it must be one of ['walk', 'stand', 'left', 'right']
        """

        self.action_mapper[action]()
        self.distance_calculator.update_distance(self._driver.current_url)
        logging.info("You have made {} meters".format(self.distance_calculator.distance))


class DistanceMeter:
    """This class is used to keep track of the distance made during a walk.
    It retrieves coordinates from a google maps url using regex and calculates
    distance walked.
    """

    def __init__(self, starting_url=None, starting_coords=None, units='km'):
        self.distance = 0
        self.coordinate_finder = r"(?:(?<=@))(?:-{1})?[0-9]+.[0-9]+,(?:-{1})?[0-9]+.[0-9]+"
        if starting_url is None:
            self.last_coords = starting_coords
        else:
            self.last_coords = self.retrieve_coords(starting_url)

    def retrieve_coords(self, url):
        """From a Google Maps URL it obtains the coordinates
        
        Args:
            url (str): URL from which coordinates have to be extracted
        
        Returns:
            list: Coordinates in decimal format (lat, long)
        """
        return [float(x) for x in re.search(self.coordinate_finder, url).group(0).split(",")]

    def update_distance(self, url):
        """self.distance keeps record of all the distance made in the walk.
        This function updates this value.
        
        Args:
            url (str): URL of the new step
        """
        new_coords = self.retrieve_coords(url)
        self.distance += DistanceMeter.distance_from_coords(new_coords, self.last_coords)
        self.last_coords = new_coords

    @staticmethod
    def distance_from_coords(coords1, coords2):
        """Gets two pairs of coordinates in decimal format and calculates distance
        between points
        
        Args:
            coords1 (list): Coordinates in decimal format (lat, long)
            coords2 (list): Coordinates in decimal format (lat, long)
        
        Returns:
            float: Meters between points
        """
        R = 6371000
        phi1 = coords1[0] * math.pi / 180
        phi2 = coords2[0] * math.pi / 180
        deltaPhi = phi2 - phi1
        deltaLambda = (coords2[1] - coords1[1]) * math.pi / 180

        a = math.sin(deltaPhi / 2) ** 2 + (math.cos(phi1) * math.cos(phi2) * math.sin(deltaLambda / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return float(R * c)
