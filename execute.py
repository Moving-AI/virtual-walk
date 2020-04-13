from source.webcam_predictor import WebcamPredictor

if __name__ == '__main__':
    zaragoza = [41.6423746, -0.8862972]
    paris = [48.8574788, 2.3515892]
    roma = [41.9045284, 12.4941586]
    barcelona = [41.3921363, 2.1618873]
    nyc = [40.7576611, -73.9866615]
    sidney = [-33.8643791, 151.2127931]
    nueva_zelanda = [-41.2846378, 174.7772473]

    wp = WebcamPredictor(coordinates=nyc)
    wp.predictor()
