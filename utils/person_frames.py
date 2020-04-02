import math

import numpy as np

from utils.person import Person


class PersonMovement:
    def __init__(self, list_persons, times_v=10, joints_remove=(13, 14, 15, 16)):
        self.list_persons = list_persons
        self.n_frames = len(list_persons)
        self.coords = self.get_vector(times_v, joints_remove)

    def get_vector(self, times_v, joints_remove):
        '''
        Get coordinates vector from a series of frames.
        :param times_v: int. Times the body velocity is repeated in the resulting vector.
        :param joints_remove: tuple. Joints that will be removed and not used in the final vector
        :return: ndarray (1, N). Flattened vector of [x + v * times_v + v] dimensions. Where x and v are the flattened
        vectors of joints positions and velocities.
        '''
        xs = np.array([person.keypoints_positions for person in self.list_persons])
        hs = np.array([person.H for person in self.list_persons])
        ws = np.array([person.W for person in self.list_persons])

        xs = np.delete(xs, joints_remove, axis=1)
        avg_h = np.mean(hs)
        avg_w = np.mean(ws)
        x = np.empty(xs.shape)
        x[:,:,0] = (xs[:,:,0] - np.mean(xs[:,:,0])) / avg_h
        x[:,:,1] = (xs[:,:,1] - np.mean(xs[:,:,1])) / avg_w

        # Body velocity = neck velocity (index 17 before removing)
        v = np.empty(times_v - 1)
        for i_person in range(1, self.n_frames):
            vix = (x[i_person, 17 - len(joints_remove), 0] - x[i_person - 1, 17 - len(joints_remove), 0]) ** 2
            viy = (x[i_person, 17 - len(joints_remove), 1] - x[i_person - 1, 17 - len(joints_remove), 1]) ** 2
            v.append(math.sqrt(vix + viy))

        v_joints = np.empty((self.n_frames - 1, x.shape[1], x.shape[2]))
        for i_person in range(xs.shape[0] - 1):
            v_joints[i_person,:] = x[i_person + 1, :] - x[i_person, :]

        coords = np.concatenate((xs.flatten(), np.repeat(v.flatten(), times_v), v_joints.flatten()))
        coords = np.reshape(coords, (1, coords.shape[0]))

        return coords

    def write_to_txt(self, path, label):
        writer = np.append([label], self.coords)
        np.savetxt(path, writer, delimiter='\t')


if __name__ == '__main__':
    path = '../prueba.txt'
    p = Person(path_txt=path)
    p2 = p
    list_p = [p, p, p, p, p]
    group = PersonMovement(list_p)
    # c = group.get_vector(10)