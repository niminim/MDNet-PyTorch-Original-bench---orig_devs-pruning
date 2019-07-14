import os
import sys
import platform


# --------------- opencv options --------------------
# Illumination Variation - the illumination in the target region is significantly changed.
OTB_IL = ['Basketball', 'Box', 'Car1', 'Car2', 'Car24', 'Car4', 'CarDark', 'Coke', 'Crowds', 'David', 'Doll', 'FaceOcc2', 'Fish', 'Human2', 'Human4', 'Human7', 'Human8', 'Human9', 'Ironman', 'KiteSurf', 'Lemming', 'Liquor', 'Man', 'Matrix', 'Mhyang', 'MotorRolling', 'Shaking', 'Singer1', 'Singer2', 'Skating1', 'Skiing', 'Soccer', 'Sylvester', 'Tiger1', 'Tiger2', 'Trans', 'Trellis', 'Woman']

# Scale Variation – the ratio of the bounding boxes of the first frame and the current frame is out of the range ts, ts > 1 (ts=2).
OTB_SV = ['Biker', 'BlurBody', 'BlurCar2', 'BlurOwl', 'Board', 'Box', 'Boy', 'Car1', 'Car24', 'Car4', 'CarScale', 'ClifBar', 'Couple', 'Crossing', 'Dancer', 'David', 'Diving', 'Dog', 'Dog1', 'Doll', 'DragonBaby', 'Dudek', 'FleetFace', 'Freeman1', 'Freeman3', 'Freeman4', 'Girl', 'Girl2', 'Gym', 'Human2', 'Human3', 'Human4', 'Human5', 'Human6', 'Human7', 'Human8', 'Human9', 'Ironman', 'Jump', 'Lemming', 'Liquor', 'Matrix', 'MotorRolling', 'Panda', 'RedTeam', 'Rubik', 'Shaking', 'Singer1', 'Skater', 'Skater2', 'Skating1', 'Skating2', 'Skiing', 'Soccer', 'Surfer', 'Toy', 'Trans', 'Trellis', 'Twinnings', 'Vase', 'Walking', 'Walking2', 'Woman']

# Occlusion – the target is partially or fully occluded.
OTB_OCC = ['Basketball', 'Biker', 'Bird2', 'Bolt', 'Box', 'CarScale', 'ClifBar', 'Coke', 'Coupon', 'David', 'David3', 'Doll', 'DragonBaby', 'Dudek', 'FaceOcc1', 'FaceOcc2', 'Football', 'Freeman4', 'Girl', 'Girl2', 'Human3', 'Human4', 'Human5', 'Human6', 'Human7', 'Ironman', 'Jogging', 'Jump', 'KiteSurf', 'Lemming', 'Liquor', 'Matrix', 'Panda', 'RedTeam', 'Rubik', 'Singer1', 'Skating1', 'Skating2', 'Soccer', 'Subway', 'Suv', 'Tiger1', 'Tiger2', 'Trans', 'Walking', 'Walking2', 'Woman']

# Deformation – non-rigid object deformation.
OTB_DEF = ['Basketball', 'Bird1', 'Bird2', 'BlurBody', 'Bolt', 'Bolt2', 'Couple', 'Crossing', 'Crowds', 'Dancer', 'Dancer2', 'David', 'David3', 'Diving', 'Dog', 'Dudek', 'FleetFace', 'Girl2', 'Gym', 'Human3', 'Human4', 'Human5', 'Human6', 'Human7', 'Human8', 'Human9', 'Jogging', 'Jump', 'Mhyang', 'Panda', 'Singer2', 'Skater', 'Skater2', 'Skating1', 'Skating2', 'Skiing', 'Subway', 'Tiger1', 'Tiger2', 'Trans', 'Walking', 'Woman']

# Motion Blur – the target region is blurred due to the motion of target or camera.
OTB_MB = ['Biker', 'BlurBody', 'BlurCar1', 'BlurCar2', 'BlurCar3', 'BlurCar4', 'BlurFace', 'BlurOwl', 'Board', 'Box', 'Boy', 'ClifBar', 'David', 'Deer', 'DragonBaby', 'FleetFace', 'Girl2', 'Human2', 'Human7', 'Human9', 'Ironman', 'Jump', 'Jumping', 'Liquor', 'MotorRolling', 'Soccer', 'Tiger1', 'Tiger2', 'Woman']

# Fast Motion – the motion of the ground truth is larger than tm pixels (tm=20).
OTB_FM = ['Biker', 'Bird1', 'Bird2', 'BlurBody', 'BlurCar1', 'BlurCar2', 'BlurCar3', 'BlurCar4', 'BlurFace', 'BlurOwl', 'Board', 'Boy', 'CarScale', 'ClifBar', 'Coke', 'Couple', 'Deer', 'DragonBaby', 'Dudek', 'FleetFace', 'Human6', 'Human7', 'Human9', 'Ironman', 'Jumping', 'Lemming', 'Liquor', 'Matrix', 'MotorRolling', 'Skater2', 'Skating2', 'Soccer', 'Surfer', 'Tiger1', 'Tiger2', 'Toy', 'Vase', 'Woman']

# In-Plane Rotation – the target rotates in the image plane.
OTB_IPR = ['Bird2', 'BlurBody', 'BlurFace', 'BlurOwl', 'Bolt', 'Box', 'Boy', 'CarScale', 'ClifBar', 'Coke', 'Dancer', 'David', 'David2', 'Deer', 'Diving', 'Dog1', 'Doll', 'DragonBaby', 'Dudek', 'FaceOcc2', 'FleetFace', 'Football', 'Football1', 'Freeman1', 'Freeman3', 'Freeman4', 'Girl', 'Gym', 'Ironman', 'Jump', 'KiteSurf', 'Matrix', 'MotorRolling', 'MountainBike', 'Panda', 'RedTeam', 'Rubik', 'Shaking', 'Singer2', 'Skater', 'Skater2', 'Skiing', 'Soccer', 'Surfer', 'Suv', 'Sylvester', 'Tiger1', 'Tiger2', 'Toy', 'Trellis', 'Vase']

# Out-of-Plane Rotation – the target rotates out of the image plane.
OTB_OPR = ['Basketball', 'Biker', 'Bird2', 'Board', 'Bolt', 'Box', 'Boy', 'CarScale', 'Coke', 'Couple', 'Dancer', 'David', 'David2', 'David3', 'Dog', 'Dog1', 'Doll', 'DragonBaby', 'Dudek', 'FaceOcc2', 'FleetFace', 'Football', 'Football1', 'Freeman1', 'Freeman3', 'Freeman4', 'Girl', 'Girl2', 'Gym', 'Human2', 'Human3', 'Human6', 'Ironman', 'Jogging', 'Jump', 'KiteSurf', 'Lemming', 'Liquor', 'Matrix', 'Mhyang', 'MountainBike', 'Panda', 'RedTeam', 'Rubik', 'Shaking', 'Singer1', 'Singer2', 'Skater', 'Skater2', 'Skating1', 'Skating2', 'Skiing', 'Soccer', 'Surfer', 'Sylvester', 'Tiger1', 'Tiger2', 'Toy', 'Trellis', 'Twinnings', 'Woman']

# Out-of-View – some portion of the target leaves the view.
OTB_OV = ['Biker', 'Bird1', 'Board', 'Box', 'ClifBar', 'DragonBaby', 'Dudek', 'Human6', 'Ironman', 'Lemming', 'Liquor', 'Panda', 'Suv', 'Tiger2']

# Background Clutters – the background near the target has the similar color or texture as the target.
OTB_BC = ['Basketball', 'Board', 'Bolt2', 'Box', 'Car1', 'Car2', 'Car24', 'CarDark', 'ClifBar', 'Couple', 'Coupon', 'Crossing', 'Crowds', 'David3', 'Deer', 'Dudek', 'Football', 'Football1', 'Human3', 'Ironman', 'Liquor', 'Matrix', 'Mhyang', 'MotorRolling', 'MountainBike', 'Shaking', 'Singer2', 'Skating1', 'Soccer', 'Subway', 'Trellis']

# Low Resolution – the number of pixels inside the ground-truth bounding box is less than tr (tr =400).
OTB_LR = ['Biker', 'Car1', 'Freeman3', 'Freeman4', 'Panda', 'RedTeam', 'Skiing', 'Surfer', 'Walking']

OTB_attributes_dict = {'IL': OTB_IL, 'SV': OTB_SV, 'OCC':OTB_OCC, 'DEF':OTB_DEF, 'MB':OTB_MB, 'FM':OTB_FM,
                       'IPR':OTB_IPR, 'OPR':OTB_OPR, 'OV':OTB_OV, 'BC':OTB_BC, 'LR':OTB_LR}

OTB_select_attributes_strings = ['IL', 'SV', 'OCC', 'FM', 'BC', 'LR']
# ---------------------------------------------------



# seq_home = '../dataset/'
usr_home = os.path.expanduser('~')
#OS = platform.system()
#if OS == 'Windows':
    # usr_home = 'C:/Users/smush/'
#    seq_home = os.path.join(usr_home, 'downloads/')
#elif OS == 'Linux':
#    # usr_home = '~/'
#    #seq_home = os.path.join(usr_home, 'MDNet-data/')
#    seq_home = os.path.join(usr_home, '/Downloads/datasets/')
#else:
#    sys.exit("aa! errors!")
#seq_home = os.path.join(usr_home, '/Downloads/datasets/')    
seq_home = 'datasets'

#seq_home = os.path.join(usr_home, 'Downloads/datasets/')


benchmark_dataset = 'VOT/vot2016'
#benchmark_dataset = 'OTB/otb100'
seq_home = os.path.join(seq_home, benchmark_dataset)


my_sequence_list = ['DragonBaby']
#my_sequence_list = ['DragonBaby', 'Bird1']
#my_sequence_list = [ 'Walking', 'Walking2', 'Woman']

# cd Desktop/NN/MDNet-PyTorch-Original-bench\ -\ orig_devs-pruning
# python tracking/run_tracker.py -tr -f -p

#my_sequence_list = ['Basketball', 'Biker', 'Bird2', ]

#my_sequence_list = ['DragonBaby', 'Bird1']  #, 'Car4', 'BlurFace']

# OTB-50 list
#my_sequence_list = [' Basketball', 'Biker', 'Bird1', 'BlurBody', 'BlurCar', 'BlucCar2', 'BlurOwl', 'Bolt', 'Box', 'Car1', 'Car4', 'CarDark', 'CarScale', 
#                    'ClifBar', 'Couple', 'Crowds','David', 'Deer','Diving', 'DragonBaby', 'Dudek', 'Football', 'Freeman4', 'Girl', 'Human3', 'Human4',
#                    'Human6', 'Human9', 'Ironman', 'Jump', 'Jumping', 'Liqour', 'Matrix', 'MotorRolling', 'Panda', 'RedTeam', 'Shaking', 'Singer2',
#                    'Skating1', 'Skating2', 'Skiing', 'Soccer', 'Surfer', 'Sylvester', 'Tiger2', 'Trellis', 'Walking', 'Walking2', 'Woman'] 


# VOT-2016
my_sequence_list = ['bag',        'ball1',       'ball2',      'basketball',   'birds1',     'birds2',   'blanket',     'bmx',          'bolt1',        'bolt2',      'book',     'butterfly', 'car1', 
                    'car2',        'crossing',    'dinosaur',   'fernando',     'fish1',      'fish2',    'fish3',       'fish4',        'girl',         'glove',     'godfather', 'graduate',
                    'gymnastics1', 'gymnastics2', 'gymnastics3','gymnastics4',  'hand',       'handball1', 'handball2',  'helicopter',   'iceskater1',   'iceskater2', 
                    'leaves',      'marching',    'matrix',     'motocross1',   'motocross2', 'nature',    'octopus',    'pedestrian1',  'pedestrian2',  'rabbit',     'racing',  'road',
                    'shaking',     'sheep',       'singer1',    'singer2',      'singer3',    'soccer1',   'soccer2',    'soldier',      'sphere',       'tiger',     'traffic',   'tunnel', 'wiper'] 


show_average_over_sequences = True
show_per_sequence = True

# benchmarking
losses_strings = {1: 'BCE'}
# display_benchmark_results = True
avg_iters_per_sequence = 1  # should be 15 per the VOT challenge
models_indices_for_tracking = [1]  # single model we trained offline for testing
models_strings = {1: 'orig-devs-pruning'}  # we work on the original model that was trained by the researcher
models_paths = {1: None}
loss_indices_for_tracking = [1]  # single loss version we test
sequence_len_limit = 10
detailed_printing = False

init_after_loss = False  # True - VOT metrics, False - OTB metrics
display_VOT_benchmark = True
display_OTB_benchmark = True



