import numpy as np
import random
import eval7

class OddsAgentV20(object):
    ''' A random agent. Random agents is for running toy examples on the card games
    '''

    def __init__(self, num_actions, threshold_offset={
            'preflop': 0
            ,'flop'  : 0
            ,'turn'  : 0
            ,'river' : 0
            }):
        ''' Initilize the random agent

        Args:
            num_actions (int): The size of the ouput action space
        '''
        self.use_raw = True
        self.num_actions = num_actions
        
        # Pre-calculated
        self.preflop_value_dict = {
             '32o': 0.0,
             '42o': 0.0059523809523809,
             '52o': 0.0119047619047619,
             '62o': 0.0178571428571428,
             '72o': 0.0238095238095238,
             '32s': 0.0297619047619047,
             '43o': 0.0357142857142857,
             '63o': 0.0416666666666666,
             '42s': 0.0476190476190476,
             '53o': 0.0535714285714285,
             '62s': 0.0595238095238095,
             '73o': 0.0654761904761904,
             '82o': 0.0714285714285714,
             '72s': 0.0773809523809523,
             '52s': 0.0833333333333333,
             '54o': 0.0892857142857142,
             '43s': 0.0952380952380952,
             '83o': 0.1011904761904761,
             '64o': 0.1071428571428571,
             '74o': 0.1130952380952381,
             '53s': 0.119047619047619,
             '63s': 0.125,
             '92o': 0.1309523809523809,
             '84o': 0.1369047619047619,
             '82s': 0.1428571428571428,
             '65o': 0.1488095238095238,
             '73s': 0.1547619047619047,
             '83s': 0.1607142857142857,
             '93o': 0.1666666666666666,
             '75o': 0.1726190476190476,
             '54s': 0.1785714285714285,
             '94o': 0.1845238095238095,
             '64s': 0.1904761904761904,
             'T2o': 0.1964285714285714,
             '85o': 0.2023809523809523,
             '74s': 0.2083333333333333,
             '76o': 0.2142857142857142,
             '84s': 0.2202380952380952,
             '92s': 0.2261904761904762,
             'T3o': 0.2321428571428571,
             '95o': 0.238095238095238,
             '65s': 0.244047619047619,
             '94s': 0.25,
             '93s': 0.2559523809523809,
             'T4o': 0.2619047619047619,
             '86o': 0.2678571428571428,
             '75s': 0.2738095238095238,
             '76s': 0.2797619047619047,
             '96o': 0.2857142857142857,
             'T3s': 0.2916666666666667,
             'T2s': 0.2976190476190476,
             '86s': 0.3035714285714285,
             '87o': 0.3095238095238095,
             'T5o': 0.3154761904761904,
             '85s': 0.3214285714285714,
             '95s': 0.3273809523809524,
             'J2o': 0.3333333333333333,
             'J3o': 0.3392857142857143,
             'J2s': 0.3452380952380952,
             '97o': 0.3511904761904761,
             'T4s': 0.3571428571428571,
             'J4o': 0.3630952380952381,
             '22': 0.369047619047619,
             'T6o': 0.375,
             '96s': 0.3809523809523809,
             'Q2o': 0.3869047619047619,
             'J5o': 0.3928571428571428,
             'J6o': 0.3988095238095238,
             '98o': 0.4047619047619047,
             '87s': 0.4107142857142857,
             'Q3o': 0.4166666666666667,
             'T7o': 0.4226190476190476,
             'T6s': 0.4285714285714285,
             'J4s': 0.4345238095238095,
             'J3s': 0.4404761904761904,
             'T5s': 0.4464285714285714,
             '97s': 0.4523809523809524,
             'J5s': 0.4583333333333333,
             'Q4o': 0.4642857142857143,
             'Q5o': 0.4702380952380952,
             'T8o': 0.4761904761904761,
             'J7o': 0.4821428571428571,
             '33': 0.4880952380952381,
             'Q2s': 0.494047619047619,
             'K2o': 0.5,
             'Q3s': 0.5059523809523809,
             'J6s': 0.5119047619047619,
             '98s': 0.5178571428571429,
             'T7s': 0.5238095238095238,
             'Q6o': 0.5297619047619048,
             'K3o': 0.5357142857142857,
             'T9o': 0.5416666666666666,
             'J8o': 0.5476190476190477,
             'T8s': 0.5535714285714286,
             'Q7o': 0.5595238095238095,
             'Q5s': 0.5654761904761905,
             'Q4s': 0.5714285714285714,
             'J7s': 0.5773809523809523,
             'Q6s': 0.5833333333333334,
             'K2s': 0.5892857142857143,
             'K4o': 0.5952380952380952,
             'T9s': 0.6011904761904762,
             '44': 0.6071428571428571,
             'K3s': 0.6130952380952381,
             'J9o': 0.6190476190476191,
             'K5o': 0.625,
             'Q8o': 0.6309523809523809,
             'Q7s': 0.6369047619047619,
             'J8s': 0.6428571428571429,
             'K4s': 0.6488095238095238,
             'K6o': 0.6547619047619048,
             'A2o': 0.6607142857142857,
             'Q8s': 0.6666666666666666,
             'K5s': 0.6726190476190477,
             'JTo': 0.6785714285714286,
             'J9s': 0.6845238095238095,
             'K7o': 0.6904761904761905,
             'Q9o': 0.6964285714285714,
             'A3o': 0.7023809523809523,
             'K6s': 0.7083333333333334,
             'A2s': 0.7142857142857143,
             'K8o': 0.7202380952380952,
             '55': 0.7261904761904762,
             'JTs': 0.7321428571428571,
             'A4o': 0.7380952380952381,
             'QTo': 0.7440476190476191,
             'K7s': 0.75,
             'Q9s': 0.7559523809523809,
             'K8s': 0.7619047619047619,
             'A3s': 0.7678571428571429,
             'QJo': 0.7738095238095238,
             'A6o': 0.7797619047619048,
             'A5o': 0.7857142857142857,
             'K9o': 0.7916666666666666,
             'A4s': 0.7976190476190477,
             'A7o': 0.8035714285714286,
             'QTs': 0.8095238095238095,
             'A6s': 0.8154761904761905,
             '66': 0.8214285714285714,
             'K9s': 0.8273809523809523,
             'A5s': 0.8333333333333334,
             'KTo': 0.8392857142857143,
             'QJs': 0.8452380952380952,
             'A8o': 0.8511904761904762,
             'KJo': 0.8571428571428571,
             'A9o': 0.8630952380952381,
             'A8s': 0.8690476190476191,
             'KTs': 0.875,
             '77': 0.8809523809523809,
             'KQo': 0.8869047619047619,
             'A7s': 0.8928571428571429,
             'KJs': 0.8988095238095238,
             'A9s': 0.9047619047619048,
             'ATo': 0.9107142857142856,
             'AJo': 0.9166666666666666,
             'KQs': 0.9226190476190476,
             'ATs': 0.9285714285714286,
             'AQo': 0.9345238095238096,
             'AJs': 0.9404761904761904,
             '88': 0.9464285714285714,
             'AKo': 0.9523809523809524,
             'AQs': 0.9583333333333334,
             '99': 0.9642857142857144,
             'AKs': 0.9702380952380952,
             'TT': 0.9761904761904762,
             'JJ': 0.9821428571428572,
             'QQ': 0.988095238095238,
             'KK': 0.9940476190476192,
             'AA': 1.0
         }
        
        # Parameters
        self.para_all = {
           'preflop':{
               # threshold in hand strength
               'threshold_re_rasise'    : 0.80 + threshold_offset['preflop']
               ,'threshold_open_raise'  : 0.70 + threshold_offset['preflop']
               ,'threshold_call'        : 0.60 + threshold_offset['preflop']
               #  add threshold to re_raise multiply times
               ,'threshold_re_raise_add': 0.05
               # mixing
               ,'pct_raise_mix_call'    : 0.30 # if other raised
               ,'pct_raise_mix_check'   : 0.10 # if no one raised
               ,'pct_call_mix_raise'    : 0.25 # others must raised first so that i can call
               ,'pct_call_mix_fold'     : 0.15 # others must raised first so that i can call 
               ,'pct_fold_mix_call'     : 0.05 # others must raised first so that i can call
               ,'pct_fold_mix_raise'    : 0.10 # we can bluff a bit...
               ,'pct_check_mix_raise'   : 0.10 # we can bluff a bit...
               }
           ,'flop':{
               # threshold in hand strength
               'threshold_re_rasise'    : 0.80 + threshold_offset['flop']
               ,'threshold_open_raise'  : 0.70 + threshold_offset['flop']
               ,'threshold_call'        : 0.60 + threshold_offset['flop']
               #  add threshold to re_raise multiply times
               ,'threshold_re_raise_add': 0.05
               # mixing
               ,'pct_raise_mix_call'    : 0.30 # if other raised
               ,'pct_raise_mix_check'   : 0.10 # if no one raised
               ,'pct_call_mix_raise'    : 0.25 # others must raised first so that i can call
               ,'pct_call_mix_fold'     : 0.15 # others must raised first so that i can call 
               ,'pct_fold_mix_call'     : 0.05 # others must raised first so that i can call
               ,'pct_fold_mix_raise'    : 0.10 # we can bluff a bit...
               ,'pct_check_mix_raise'   : 0.10 # we can bluff a bit...
              }
           ,'turn':{
               # threshold in hand strength
               'threshold_re_rasise'    : 0.80 + threshold_offset['turn']
               ,'threshold_open_raise'  : 0.70 + threshold_offset['turn']
               ,'threshold_call'        : 0.60 + threshold_offset['turn']
               #  add threshold to re_raise multiply times
               ,'threshold_re_raise_add': 0.05
               # mixing
               ,'pct_raise_mix_call'    : 0.30 # if other raised
               ,'pct_raise_mix_check'   : 0.10 # if no one raised
               ,'pct_call_mix_raise'    : 0.25 # others must raised first so that i can call
               ,'pct_call_mix_fold'     : 0.15 # others must raised first so that i can call 
               ,'pct_fold_mix_call'     : 0.05 # others must raised first so that i can call
               ,'pct_fold_mix_raise'    : 0.10 # we can bluff a bit...
               ,'pct_check_mix_raise'   : 0.10 # we can bluff a bit...
              }
           ,'river':{
               # threshold in hand strength
               'threshold_re_rasise'    : 0.80 + threshold_offset['river']
               ,'threshold_open_raise'  : 0.70 + threshold_offset['river']
               ,'threshold_call'        : 0.60 + threshold_offset['river']
               #  add threshold to re_raise multiply times
               ,'threshold_re_raise_add': 0.05
               # mixing
               ,'pct_raise_mix_call'    : 0.30 # if other raised
               ,'pct_raise_mix_check'   : 0.10 # if no one raised
               ,'pct_call_mix_raise'    : 0.25 # others must raised first so that i can call
               ,'pct_call_mix_fold'     : 0.15 # others must raised first so that i can call 
               ,'pct_fold_mix_call'     : 0.05 # others must raised first so that i can call
               ,'pct_fold_mix_raise'    : 0.10 # we can bluff a bit...
               ,'pct_check_mix_raise'   : 0.10 # we can bluff a bit...
              }
           }
        
        
        # ranges at different participation: https://www.splitsuit.com/poker-ranges-reading
        self.range_full = eval7.HandRange(
            "A2+, K2+, Q2+, J2+, T2+, 92+, 82+, 72+, 62+, 52+, 42+, 32+, 22+")
        self.range_50 = eval7.HandRange(
            "22+, A2s+, K2s+, Q7s+, J7s+, T7s+, 96s+, 86s+, 75s+, 64s+, 53s+, 43s, A2o+, K5o+, Q8o+, J8o+, T8o+, 98o, 87o, 76o, 65o")
        self.range_35 = eval7.HandRange(
            "22+, A2s+, K8s+, Q8s+, J8s+, T7s+, 97s+, 86s+, 75s+, 64s+, 54s, 43s, A8o+, A5o-A2o, K9o+, Q9o+, J9o+, T9o")
        self.range_25 = eval7.HandRange(
            "22+, A7s+, K9s+, Q9s+, J9s+, T8s+, 97s+, 86s+, 75s+, 64s+, 54s, A9o+, KTo+, QTo+, JTo, T9o")
        self.range_15 = eval7.HandRange(
            "22+,  ATs+,  KJs+,  QJs,  JTs,  T9s,  98s,  87s,  76s,  65s,  AJo+,  KJo+,  QJo")
        
    
    def step_generic_para(self, state, step_no):
        
        # step_no = 2
        step_name = {
            0: 'preflop'
            ,1: 'flop'
            ,2: 'turn'
            ,3: 'river'
            }[step_no]
        para_step = self.para_all[step_name]
        
        
        ## calculate useful information
        # 1. hand strength
        h1, h2 = state['raw_obs']['hand']
        if h1[1] == h2[1]:
            hand_169 = h1[1]+h2[1]
        elif h1[0] == h2[0]:
            hand_169 = h1[1]+h2[1]+'s'
        else:
            hand_169 = h1[1]+h2[1]+'o'
        if self.preflop_value_dict.get(hand_169):
            my_strength = self.preflop_value_dict.get(hand_169)
        else:
            my_strength = 0
        
        # 2. equity, if we have deck card
        if step_no >= 1:
            state_hand_eval         = list(map(lambda x: eval7.Card(x[1]+x[0].lower()), state['raw_obs']['hand']))
            state_public_cards_eval = list(map(lambda x: eval7.Card(x[1]+x[0].lower()), state['raw_obs']['public_cards']))
            
            # evaluate ranges
            hand_equity_full = eval7.py_hand_vs_range_exact(state_hand_eval, self.range_full, state_public_cards_eval)
            hand_equity_50   = eval7.py_hand_vs_range_exact(state_hand_eval, self.range_50  , state_public_cards_eval)
            hand_equity_35   = eval7.py_hand_vs_range_exact(state_hand_eval, self.range_35  , state_public_cards_eval)
            # hand_equity_25   = eval7.py_hand_vs_range_exact(state_hand_eval, self.range_25  , state_public_cards_eval)
            
            # wider-range at flop
            if step_name == 'flop':
                my_equity = hand_equity_full
            
            # narrow-range at turn
            elif step_name == 'turn':
                my_equity = hand_equity_50
            
            # most-narrow range at river
            else:
                my_equity = hand_equity_35
        else:
            my_equity = 0.6
        
        
        ## Pick an value for decision making
        decision_value = my_strength if step_no == 0 else my_equity
        
        ### Start Decision Rules
        ## If other ppl didn't raise into me (i.e. either i am first to act, or all other ppl checked)
        if 'check' in state['raw_legal_actions']:
            
            # (1) if our hand is good, open raise if possible else call. 
            #   Mix raise with check
            if decision_value > para_step['threshold_open_raise']:
                
                #  raise mix check 
                if random.uniform(0, 1) < para_step['pct_raise_mix_check']:
                    return 'check'  
                #  mostly just raise
                else:
                    # there is a re-raise limit, if we hits that just call
                    return 'raise' if 'raise' in state['raw_legal_actions'] else 'check'
            # (2) our hand is not good, mostly just check
            else:
                
                ##  check mix raise 
                if random.uniform(0, 1) < para_step['pct_check_mix_raise']:
                    return 'raise' if 'raise' in state['raw_legal_actions'] else 'check'
                # mostly just check
                else:
                    return 'check'  
            
            
        ## If other ppl raised into me
        else:
            # if no check, must be due to other ppl raised, must have 'call' as an option
            assert 'call' in state['raw_legal_actions']
            
            # count how many raises there are in current round, add that to the threshold
            raise_nums_step = state['raw_obs']['raise_nums'][step_no]
            raise_nums_step_add_threshold = (raise_nums_step-1) * para_step['threshold_re_raise_add']
            
            # (1) if our hand is really good, re-raise if possible else call. 
            #   Mix raise with call
            if decision_value > (para_step['threshold_re_rasise'] + raise_nums_step_add_threshold):
                
                #  raise mix call 
                if random.uniform(0, 1) < para_step['pct_raise_mix_call']:
                    return 'call'  # 'call' is definitely an option since ppl raised into me 
                #  mostly just raise
                else:
                    # there is a re-raise limit, if we hits that just call
                    return 'raise' if 'raise' in state['raw_legal_actions'] else 'call'
            
            # (2) if our hand is so-so, we can call a raise
            #    Mix call with raise and fold
            elif decision_value > (para_step['threshold_call'] + raise_nums_step_add_threshold):
                
                ##  call mix raise 
                if random.uniform(0, 1) < para_step['pct_call_mix_raise']:
                    # there is a re-raise limit, if we hits that just call
                    return 'raise' if 'raise' in state['raw_legal_actions'] else 'call'
                # mostly just call
                else:
                    ##  call mix fold 
                    if random.uniform(0, 1) < (para_step['pct_call_mix_fold']/(1-para_step['pct_call_mix_raise'])):
                        # there is a re-raise limit, if we hits that just call
                        return 'fold' # you can always fold
                    # mostly just call
                    else:
                        return 'call'  # 'call' is definitely an option since ppl raised into me 
                
            # (3) if our hand is quit bad, we mostly fold
            #    Mix fold with raise and call
            else:
            
                ##  fold mix call 
                if random.uniform(0, 1) < para_step['pct_fold_mix_call']:
                    return 'call' 
                
                # mostly just fold
                else:
                    ##  fold mix raise 
                    if random.uniform(0, 1) < (para_step['pct_fold_mix_raise']/(1-para_step['pct_fold_mix_call'])):
                        return 'raise' if 'raise' in state['raw_legal_actions'] else 'call'
                    
                    # mostly just fold
                    else:
                        return 'fold' 
                    
        
    
    def step(self, state):
        ''' Predict the action given the curent state in gerenerating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
        '''
        
        # PREFLOP => just look at pre-flop values
        if len(state['raw_obs']['public_cards']) == 0:
            return self.step_generic_para(state, 0)
        ## FLOP
        elif len(state['raw_obs']['public_cards']) == 3:
            return self.step_generic_para(state, 0)
        ## TURN
        elif len(state['raw_obs']['public_cards']) == 4:
            return self.step_generic_para(state, 0)
        ## RIVER
        else:
            return self.step_generic_para(state, 0)
            
        

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
        probs = [0 for _ in range(self.num_actions)]
        for i in state['legal_actions']:
            probs[i] = 1/len(state['legal_actions'])

        info = {}
        info['probs'] = {state['raw_legal_actions'][i]: probs[list(state['legal_actions'].keys())[i]] for i in range(len(state['legal_actions']))}

        return self.step(state), info
