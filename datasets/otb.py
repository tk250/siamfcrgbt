from __future__ import absolute_import, print_function, unicode_literals

import os
import glob
import numpy as np
import io
import six
from itertools import chain
import ops

from ..utils.ioutils import download, extract


class OTB(object):
    r"""`OTB <http://cvlab.hanyang.ac.kr/tracker_benchmark/>`_ Datasets.

    Publication:
        ``Object Tracking Benchmark``, Y. Wu, J. Lim and M.-H. Yang, IEEE TPAMI 2015.
    
    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
        version (integer or string): Specify the benchmark version, specify as one of
            ``2013``, ``2015``, ``tb50`` and ``tb100``.
        download (boolean, optional): If True, downloads the dataset from the internet
            and puts it in root directory. If dataset is downloaded, it is not
            downloaded again.
    """
    __otb13_seqs = ['Basketball', 'Bolt', 'Boy', 'Car4', 'CarDark',
                    'CarScale', 'Coke', 'Couple', 'Crossing', 'David',
                    'David2', 'David3', 'Deer', 'Dog1', 'Doll', 'Dudek',
                    'FaceOcc1', 'FaceOcc2', 'Fish', 'FleetFace',
                    'Football', 'Football1', 'Freeman1', 'Freeman3',
                    'Freeman4', 'Girl', 'Ironman', 'Jogging', 'Jumping',
                    'Lemming', 'Liquor', 'Matrix', 'Mhyang', 'MotorRolling',
                    'MountainBike', 'Shaking', 'Singer1', 'Singer2',
                    'Skating1', 'Skiing', 'Soccer', 'Subway', 'Suv',
                    'Sylvester', 'Tiger1', 'Tiger2', 'Trellis', 'Walking',
                    'Walking2', 'Woman']

    __tb50_seqs = ['Basketball', 'Biker', 'Bird1', 'BlurBody', 'BlurCar2',
                   'BlurFace', 'BlurOwl', 'Bolt', 'Box', 'Car1', 'Car4',
                   'CarDark', 'CarScale', 'ClifBar', 'Couple', 'Crowds',
                   'David', 'Deer', 'Diving', 'DragonBaby', 'Dudek',
                   'Football', 'Freeman4', 'Girl', 'Human3', 'Human4',
                   'Human6', 'Human9', 'Ironman', 'Jump', 'Jumping',
                   'Liquor', 'Matrix', 'MotorRolling', 'Panda', 'RedTeam',
                   'Shaking', 'Singer2', 'Skating1', 'Skating2', 'Skiing',
                   'Soccer', 'Surfer', 'Sylvester', 'Tiger2', 'Trellis',
                   'Walking', 'Walking2', 'Woman']

    __tb100_seqs = ['Bird2', 'BlurCar1', 'BlurCar3', 'BlurCar4', 'Board',
                    'Bolt2', 'Boy', 'Car2', 'Car24', 'Coke', 'Coupon',
                    'Crossing', 'Dancer', 'Dancer2', 'David2', 'David3',
                    'Dog', 'Dog1', 'Doll', 'FaceOcc1', 'FaceOcc2', 'Fish',
                    'FleetFace', 'Football1', 'Freeman1', 'Freeman3',
                    'Girl2', 'Gym', 'Human2', 'Human5', 'Human7', 'Human8',
                    'Jogging', 'KiteSurf', 'Lemming', 'Man', 'Mhyang',
                    'MountainBike', 'Rubik', 'Singer1', 'Skater',
                    'Skater2', 'Subway', 'Suv', 'Tiger1', 'Toy', 'Trans',
                    'Twinnings', 'Vase'] + __tb50_seqs

    __otb15_seqs = __tb100_seqs

    __infrared = ['afterrain', 'aftertree', 'baby', 'baginhand',
                  'baketballwaliking', 'balancebike', 'basketball2',
                  'bicyclecity', 'bike', 'bikeman', 'bikemove1',
                  'biketwo', 'blackwoman', 'bluebike', 'blueCar',
                  'boundaryandfast', 'bus6', 'call', 'car', 'car3',
                  'car10', 'car20', 'car37', 'car41', 'car66', 'caraftertree', 
                  'carLight', 'carnotfar', 'carnotmove', 'carred', 'child', 'child1', 
                  'child3', 'child4', 'children2', 'children3', 'children4', 'crossroad',
                  'crouch', 'cycle1', 'cycle2', 'cycle3', 'cycle4', 'cycle5', 'diamond', 
                  'dog', 'dog1', 'dog10', 'dog11', 'elecbike', 'elecbike10', 'elecbike2', 
                  'elecbike3', 'elecbikechange2', 'elecbikeinfrontcar',  'luggage', 'man2',
                  'man22', 'man23', 'man24', 'man26', 'man28', 'man29', 'man3', 'man4', 
                  'man45', 'man5', 'man55', 'man68', 'man69', 'man7', 'man8', 'man88', 'man9',
                  'manafterrain', 'mancross', 'mancross1', 'mancrossandup', 'mandrivecar',
                  'manfaraway', 'maninblack', 'maninglass', 'maningreen2', 'maninred', 
                  'manlight', 'manoccpart', 'manonboundary', 'manonelecbike', 'manontricycle',
                  'manout2', 'manup', 'manwithbag', 'manwithbag4', 'manwithbasketball',
                  'manwithluggage', 'manwithumbrella', 'manypeople', 'manypeople1', 
                  'manypeople2', 'mobile', 'night2', 'nightcar', 'nightrun', 
                  'nightthreepeople', 'notmove', 'oldman', 'oldman2', 'oldwoman',
                  'orangeman1', 'people', 'people1', 'people3', 'playsoccer',
                  'push', 'rainingwaliking', 'raningcar', 'redbag', 'redcar', 
                  'redcar2', 'redmanchange', 'rmo', 'run', 'run1', 'run2', 'scooter',
                  'shake', 'shoeslight', 'single1', 'single3', 'soccer', 'soccer2', 
                  'soccerinhand', 'straw', 'stroller', 'supbus', 'supbus2', 'takeout',
                  'tallman', 'threeman', 'threeman2', 'threepeople', 'threewoman2',
                  'together', 'toy1', 'toy3', 'toy4', 'tree2', 'tree3', 'tree5',
                  'trees', 'tricycle', 'tricycle1', 'tricycle2', 'tricycle6', 
                  'tricycle9', 'tricyclefaraway', 'tricycletwo', 'twoelecbike',
                  'twoelecbike1', 'twoman', 'twoman1', 'twoman2', 'twoperson',
                  'twowoman', 'twowoman1', 'walking40', 'walking41', 'walkingman',
                  'walkingman1', 'walkingman12', 'walkingman20', 'walkingman41',
                  'walkingmantiny', 'walkingnight', 'walkingtogether', 
                  'walkingtogether1', 'walkingtogetherright', 'walkingwithbag1',
                  'walkingwithbag2', 'walkingwoman', 'whitebag', 'whitecar', 
                  'whitecar3', 'whitecar4', 'whitecarafterrain', 'whiteman1', 
                  'whitesuv', 'woamn46', 'woamnwithbike', 'woman', 'woman1', 
                  'woman100', 'woman2', 'woman3', 'woman4', 'woman48', 'woman6',
                  'woman89', 'woman96', 'woman99', 'womancross', 'womanfaraway',
                  'womaninblackwithbike', 'womanleft', 'womanpink', 'womanred', 
                  'womanrun', 'womanwithbag6', 'yellowcar']
    __test = ['elecbikewithhat',
              'elecbikewithlight', 'elecbikewithlight1', 'face1', 'floor-1', 'flower1',
              'flower2', 'fog', 'fog6', 'glass', 'glass2', 'graycar2', 'green', 
              'greentruck', 'greyman', 'greywoman', 'guidepost', 'hotglass', 'hotkettle',
              'inglassandmobile', 'jump', 'kettle', 'kite2', 'kite4']

    __version_dict = {
        2013: __otb13_seqs,
        2015: __otb15_seqs,
        2019: __infrared,
        'otb2013': __otb13_seqs,
        'otb2015': __otb15_seqs,
        'tb50': __tb50_seqs,
        'tb100': __tb100_seqs,
        'test': __test,
        'test_early': __test,
        'test_late': __test,
        'test_RGB': __test,
        'test_thermal': __test,
        'test_dropout': __test}

    def __init__(self, root_dir, version=2015, download=True):
        super(OTB, self).__init__()
        assert version in self.__version_dict

        self.root_dir = root_dir
        self.version = version
        if download:
            self._download(root_dir, version)
        self._check_integrity(root_dir, version)

        valid_seqs = self.__version_dict[version]
        if version in [2019, 'test', 'test_early', 'test_late', 'test_RGB', 'test_thermal', 'test_dropout']:
            self.anno_files = sorted(list(chain.from_iterable(glob.glob(
                os.path.join(root_dir, s, 'visible.txt')) for s in valid_seqs)))
        else:
            self.anno_files = sorted(list(chain.from_iterable(glob.glob(
                os.path.join(root_dir, s, 'groundtruth*.txt')) for s in valid_seqs)))
        # remove empty annotation files
        # (e.g., groundtruth_rect.1.txt of Human4)
        self.anno_files = self._filter_files(self.anno_files)
        self.seq_dirs = [os.path.dirname(f) for f in self.anno_files]
        self.seq_names = [os.path.basename(d) for d in self.seq_dirs]
        # rename repeated sequence names
        # (e.g., Jogging and Skating2)
        self.seq_names = self._rename_seqs(self.seq_names)

    def __getitem__(self, index):
        r"""        
        Args:
            index (integer or string): Index or name of a sequence.
        
        Returns:
            tuple: (img_files, anno), where ``img_files`` is a list of
                file names and ``anno`` is a N x 4 (rectangles) numpy array.
        """
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        visible_files = sorted(glob.glob(
            os.path.join(self.seq_dirs[index], 'visible/*.jpg')))

        infrared_files = sorted(glob.glob(
            os.path.join(self.seq_dirs[index], 'infrared/*.jpg')))
        

        # special sequences
        # (visit http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html for detail)
        seq_name = self.seq_names[index]
        if seq_name.lower() == 'david':
            img_files = img_files[300-1:770]
        elif seq_name.lower() == 'football1':
            img_files = img_files[:74]
        elif seq_name.lower() == 'freeman3':
            img_files = img_files[:460]
        elif seq_name.lower() == 'freeman4':
            img_files = img_files[:283]
        elif seq_name.lower() == 'diving':
            img_files = img_files[:215]

        # to deal with different delimeters
        with open(self.anno_files[index], 'r') as f:
            anno = np.loadtxt(io.StringIO(f.read().replace(',', ' ')))
        assert len(visible_files) == len(anno)
        assert anno.shape[1] == 4

        return visible_files, infrared_files, anno

    def __len__(self):
        return len(self.seq_names)

    def _filter_files(self, filenames):
        filtered_files = []
        for filename in filenames:
            with open(filename, 'r') as f:
                if f.read().strip() == '':
                    print('Warning: %s is empty.' % filename)
                else:
                    filtered_files.append(filename)

        return filtered_files

    def _rename_seqs(self, seq_names):
        # in case some sequences may have multiple targets
        renamed_seqs = []
        for i, seq_name in enumerate(seq_names):
            if seq_names.count(seq_name) == 1:
                renamed_seqs.append(seq_name)
            else:
                ind = seq_names[:i + 1].count(seq_name)
                renamed_seqs.append('%s.%d' % (seq_name, ind))

        return renamed_seqs

    def _download(self, root_dir, version):
        assert version in self.__version_dict
        seq_names = self.__version_dict[version]

        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)
        elif all([os.path.isdir(os.path.join(root_dir, s)) for s in seq_names]):
            print('Files already downloaded.')
            return

        url_fmt = 'http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/%s.zip'
        for seq_name in seq_names:
            seq_dir = os.path.join(root_dir, seq_name)
            if os.path.isdir(seq_dir):
                continue
            url = url_fmt % seq_name
            zip_file = os.path.join(root_dir, seq_name + '.zip')
            print('Downloading to %s...' % zip_file)
            download(url, zip_file)
            print('\nExtracting to %s...' % root_dir)
            extract(zip_file, root_dir)

        return root_dir

    def _check_integrity(self, root_dir, version):
        assert version in self.__version_dict
        seq_names = self.__version_dict[version]

        if os.path.isdir(root_dir) and len(os.listdir(root_dir)) > 0:
            # check each sequence folder
            for seq_name in seq_names:
                seq_dir = os.path.join(root_dir, seq_name)
                if not os.path.isdir(seq_dir):
                    print('Warning: sequence %s not exists.' % seq_name)
        else:
            # dataset not exists
            raise Exception('Dataset not found or corrupted. ' +
                            'You can use download=True to download it.')
