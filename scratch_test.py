import unittest
from unzip_scratch import unzip_scratch
import json
import sys
from scratch_parser import scratch_parser

class TestScratchParser(unittest.TestCase):
        
    
    def setUp(self):
        
        self.scratch_unzipped = unzip_scratch()
        self.prog1 = json.loads(self.scratch_unzipped.unpack_sb3("files/test.sb3"))
        self.prog2 = json.loads(self.scratch_unzipped.unpack_sb3("files/stand_check2.sb3"))
        self.prog3 = json.loads(self.scratch_unzipped.unpack_sb3("files/stand_check.sb3"))
        self.prog4 = json.loads(self.scratch_unzipped.unpack_sb3("files/infinite_two_opcode.sb3"))
        self.prog5 = json.loads(self.scratch_unzipped.unpack_sb3("files/3l_opcode.sb3"))
        self.scratch_parser_inst = scratch_parser() 
        self.maxDiff = None
        return self.shortDescription()

    def test_scratch_nothing_onclick(self):
        
        expected2 = ['event_whenflagclicked', [
                        ['looks_sayforsecs', [['SECS', '2'], ['MESSAGE', 'Hello!']]],
                        ['motion_movesteps', [['STEPS', '10']]], 
                        ['control_repeat_until', 
                            ['CONDITION', 
                                ['operator_equals', [
                                    ['OPERAND1', ['sensing_answer', [[]]], ['OPERAND2', '50']]
                                     ]
                                ],
                                ['sensing_askandwait',['QUESTION','How old are you?']]

                            ]], 
                        ['looks_thinkforsecs', [['SECS', '2'], ['MESSAGE', 'Hmm...']]]]]
        
        all_blocks_val = self.scratch_parser_inst.get_all_blocks_vals(self.prog1)
        next_val = self.scratch_parser_inst.create_next_values2(all_blocks_val)
        parsed = self.scratch_parser_inst.create_top_tree2(all_blocks_val,next_val)
        
        self.assertEqual(expected2,parsed,msg="Test failed")
             
    def test_assertEq(self):
        expected = ['a',['b',['c','d']]]
        actual = ['a',['b',['c','d']]]
        not_actual = ['a',['b',['c','e']]]
        not_not_actual = ['a',['b',['c',['e']]]]
        self.assertEqual(expected,actual,msg="Test failed")
        self.assertNotEquals(expected,not_actual,msg="Test failed")
        self.assertNotEquals(not_not_actual,not_actual,msg="Test failed")


    def test_scratch_two_arg(self):
        expected_tree1 = ['event_whenflagclicked', [['motion_movesteps', [['STEPS', '10']]], ['motion_turnright', [['DEGREES', '15']]], ['looks_sayforsecs', [['SECS', '2'], ['MESSAGE', 'Hello!']]]]]
        expected_tree2 = ['event_whenflagclicked', [['looks_thinkforsecs', [['SECS', '2'], ['MESSAGE', 'Hmm...']]], ['motion_pointindirection', [['DIRECTION', '90']]], ['motion_changexby', [['DX', '10']]]]]
        

        all_blocks_val1 = self.scratch_parser_inst.get_all_blocks_vals(self.prog2)
        all_blocks_val2 = self.scratch_parser_inst.get_all_blocks_vals(self.prog3)
        next_val1 = self.scratch_parser_inst.create_next_values2(all_blocks_val1)
        next_val2 = self.scratch_parser_inst.create_next_values2(all_blocks_val2)
        parsed1 = self.scratch_parser_inst.create_top_tree2(all_blocks_val1,next_val1)
        parsed2 = self.scratch_parser_inst.create_top_tree2(all_blocks_val2,next_val2)

        self.assertEqual(expected_tree2,parsed1,msg="Test failed")
        self.assertEqual(expected_tree1,parsed2,msg="Test failed")


    def test_infinite_loop(self):
        expected = ['event_whenflagclicked', [['looks_sayforsecs', [['SECS', '2'], ['MESSAGE', 'Infinite with two opcodes in the body']]], ['control_repeat', ['SUBSTACK', ['looks_say', [[['MESSAGE', 'opcode1']]], 'looks_think', [[['MESSAGE', 'opcode2']]]], ['MESSAGE', 'opcode1'], ['TIMES', '10']]]]]
        all_blocks_val = self.scratch_parser_inst.get_all_blocks_vals(self.prog4)
        next_val = self.scratch_parser_inst.create_next_values2(all_blocks_val)
        parsed = self.scratch_parser_inst.create_top_tree2(all_blocks_val,next_val)
        self.assertEqual(expected,parsed,msg="Test failed")
    
    def test_loop_in_a_loop_in_a_loop(self):
        expected = ['event_whenflagclicked', [['looks_sayforsecs', [['SECS', '2'], ['MESSAGE', 'loop in a loop in a loop with 2 opcodes']]], ['control_repeat', ['SUBSTACK', ['control_repeat', [['SUBSTACK', ['control_repeat', [['SUBSTACK', ['motion_movesteps', [[['STEPS', '10']]], 'sound_seteffectto', [[['VALUE', '100']]]], ['STEPS', '10'], ['TIMES', '10']]]], ['TIMES', '10']]]], ['TIMES', '10']]]]]
        
        all_blocks_val = self.scratch_parser_inst.get_all_blocks_vals(self.prog5)
        next_val = self.scratch_parser_inst.create_next_values2(all_blocks_val)
        parsed = self.scratch_parser_inst.create_top_tree2(all_blocks_val,next_val)
        self.assertEqual(expected,parsed,msg="Test failed")

    
    def tearDown(self) :
        return self.shortDescription()
        
    def test_input_blocks(self):
        expected = ['CONDITION',
                     ['operator_equals', [
                         ['OPERAND1', ['sensing_answer', [[]]],
                           ['OPERAND2', '50']
                        ]
                                        ]
                    ],
                    ['sensing_askandwait',['QUESTION','How old are you?']] 
             ]

        all_blocks_val = self.scratch_parser_inst.get_all_blocks_vals(self.prog1)
        inp_block = self.scratch_parser_inst.read_input_values_by_id(all_blocks_val,"8J%l~hNqUpt0Lfv1;iR^")
        parsed_block  = self.scratch_parser_inst.read_input_values2(all_blocks_val,inp_block)
        self.assertEqual(expected,parsed_block,msg="Test failed")

    
    def test_input_specific_keys(self):
        expected = ['SECS', ['2']]
        all_blocks = self.scratch_parser_inst.get_all_blocks_vals(self.prog1)
        block_by_id_key = self.scratch_parser_inst.get_input_block_by_id_key(all_blocks,",r([,#`OV3[DwDfw/x./","SECS")
        self.assertNotEqual(expected,block_by_id_key,msg="Test failed")

if __name__ == '__main__':
    unittest.main()
    #runner = unittest.TextTestRunner()
    #runner.run(scratch_suite())

