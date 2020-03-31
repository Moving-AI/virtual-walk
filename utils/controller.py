import time

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys


class Controller:
    def __init__(self, driver_path, initial_url, classes, time_rotation=0.5):
        '''
        Class to control the movement in Google Street View. It only works with Firefox
        :param driver_path: str. The driver to where geckodriver is located.
        :param initial_url: str. Position where the program is going to start.
        :param classes: list. Generally, ['walk', 'stand', 'left', 'right'], but the program understands any order
        :param time_rotation: The time associated to a single rotation
        '''
        self._driver = webdriver.Firefox(executable_path=driver_path)
        self._driver.set_window_position(x=-10, y=0)
        self._driver.get(initial_url)
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
        '''
        Perform an action given its index within self.classes
        :param i_action: int. Index
        '''
        action = self.classes[i_action]
        self.action_mapper[action]()

    def perform_action_name(self, action):
        '''
        Perform action given its name.
        :param action: str. Name of the action to perform. As it uses a mapper, it must be one of
        ['walk', 'stand', 'left', 'right']
        '''
        self.action_mapper[action]()

url = "https://www.google.es/maps/@41.6425054,-0.8932757,3a,86.3y,35.92h,83.74t/data=!3m6!1e1!3m4!1sB4DQl3bfNd-txTOR2bEjPg!2e0!7i16384!8i8192"
# c = Controller('../geckodriver.exe', url, ['walk', 'stand', 'left', 'right'])