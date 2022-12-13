"""
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/
Author : Benjamin Blundell - benjamin.blundell@kcl.ac.uk

result.py - Small result class for the C.elegans


"""


class Result():
    '''
    A single result from our process_testset function
    '''
    def __init__(self):
        self.source = ""
        self.og_mask = ""
        self.thresh2d = 0
        self.thresh3d = 0
        self.jacc2d = 0
        self.jacc3d = 0
        self.og_2dscore = 0
        self.new_2d_score = 0
        self.og_asi_score = 0
        self.og_asj_score = 0
        self.new_scores = []