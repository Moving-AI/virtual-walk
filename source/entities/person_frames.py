import math
import numpy as np


class PersonMovement:
    """Extracts coordinates from a list of persons. The coordinates extracted
    are the input for all the models that predict actions.

    It has two main functions:
        - get_vector: Get positions of keypoints and joint speeds. Used for Non LSTM models
        - get_vector: Get positions of keypoints. Input for LSTM model.
    
    Returns:
        PersonMovement:
    """
    def __init__(self, list_persons, times_v=10, joints_remove=(13, 14, 15, 16), model='LSTM'):
        """PersonMovement constructor.
        
        Args:
            list_persons (list): List of 5 persons to extract coordinates from.
            times_v (int, optional): Times speeds are repeated in the coordinates. Defaults to 10.
            joints_remove (tuple, optional): Joints to be removed. Keys can be found in person.py.
            Defaults to (13, 14, 15, 16).
            model (str, optional): Whether LSTM model or Feed Forward NN. Defaults to 'LSTM'.
        """
        self.list_persons = list_persons
        self.n_frames = len(list_persons)
        if model == 'LSTM':
            self.coords = self.get_vector_lstm(joints_remove)
        else:
            self.coords = self.get_vector(times_v, joints_remove)

    def get_vector(self, times_v, joints_remove):
        """Get coordinates vector from a series of frames.
        
        Args:
            times_v (int): Times the body velocity is repeated in the resulting vector.
            joints_remove (tuple): Joints that will be removed and not used in the final vector
        
        Returns:
            ndarray: Flattened vector of [x + v * times_v + v] dimensions. Where x and v are the flattened
            vectors of joints positions and velocities.
        """

        #Array of dimensions (len(list_persons), n_keypoints, 2)
        xs = np.array([person.keypoints_positions for person in self.list_persons])
        
        #Heights and widths of the people
        hs = np.array([person.H for person in self.list_persons])
        ws = np.array([person.W for person in self.list_persons])

        xs = np.delete(xs, joints_remove, axis=1)
        avg_h = np.mean(hs)
        avg_w = np.mean(ws)
        x = np.empty(xs.shape)
        x[:,:,0] = (xs[:,:,0] - np.mean(xs[:,:,0])) / avg_h
        x[:,:,1] = (xs[:,:,1] - np.mean(xs[:,:,1])) / avg_w

        # Body velocity = neck velocity (index 17 before removing)
        v = []
        for i_person in range(1, self.n_frames):
            vix = (x[i_person, 17 - len(joints_remove), 0] - x[i_person - 1, 17 - len(joints_remove), 0]) ** 2
            viy = (x[i_person, 17 - len(joints_remove), 1] - x[i_person - 1, 17 - len(joints_remove), 1]) ** 2
            v.append(math.sqrt(vix + viy))
        v = np.array(v)

        v_joints = np.empty((self.n_frames - 1, x.shape[1], x.shape[2]))
        for i_person in range(xs.shape[0] - 1):
            v_joints[i_person,:] = x[i_person + 1, :] - x[i_person, :]

        coords = np.concatenate((x.flatten(), np.repeat(v.flatten(), times_v), v_joints.flatten()))
        coords = np.reshape(coords, (1, coords.shape[0]))

        return coords

    def write_to_txt(self, path, label):
        writer = np.append([label], self.coords)
        np.savetxt(path, writer, delimiter='\t')

    def get_vector_lstm(self, joints_remove):
        """Get coordinates vector (only positions) from a series of frames.

        Args:
            joints_remove (tuple): Joints that will be removed and not used in the final vector

        Returns:
            ndarray: Flattened vector of [x] dimensions. Where x is the flattened vector of joints positions and velocities.
        """

        # Array of dimensions (len(list_persons), n_keypoints, 2)
        xs = np.array([person.keypoints_positions for person in self.list_persons])

        # Heights and widths of the people
        hs = np.array([person.H for person in self.list_persons])
        ws = np.array([person.W for person in self.list_persons])

        xs = np.delete(xs, joints_remove, axis=1)
        avg_h = np.mean(hs)
        avg_w = np.mean(ws)
        x = np.empty(xs.shape)
        x[:, :, 0] = (xs[:, :, 0] - np.mean(xs[:, :, 0])) / avg_h
        x[:, :, 1] = (xs[:, :, 1] - np.mean(xs[:, :, 1])) / avg_w

        coords = x.flatten()
        coords = np.reshape(coords, (1, self.n_frames, xs.shape[1] * 2))

        return coords
