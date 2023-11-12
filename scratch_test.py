import unittest
from unzip_scratch import unzip_scratch
import json
import sys
from scratch_parser import scratch_parser

class ScratchTester(unittest.TestCase):
        
    
    def setUp(self):
        
        self.scratch_unzipped = unzip_scratch()
        self.prog1 = json.loads(self.scratch_unzipped.unpack_sb3("files/test.sb3"))
        self.prog2 = json.loads(self.scratch_unzipped.unpack_sb3("files/stand_check2.sb3"))
        self.prog3 = json.loads(self.scratch_unzipped.unpack_sb3("files/stand_check.sb3"))
        self.prog4 = json.loads(self.scratch_unzipped.unpack_sb3("files/infinite_two_opcode.sb3"))
        self.scratch_parser_inst = scratch_parser() 
        self.maxDiff = None
        return self.shortDescription()

    def test_scratch_nothing_onclick(self):
        
        expected2 = ['event_whenflagclicked',
                        [['looks_sayforsecs', [
                                ['SECS', '2'], 
                                ['MESSAGE', 'Hello!']
                            ]
                        ],
                        ['motion_movesteps', [
                                ['STEPS', '10']

                            ]   
                        ],
                        ['control_repeat_until', 
                                ['CONDITION', 
                                    ['operator_equals', 
                                        ['OPERAND1', 
                                            ['sensing_answer', []], 
                                        ],
                                        ['OPERAND2', '50']
                                    ],
                                    ['sensing_askandwait', 
                                        ['QUESTION', 'How old are you?']
                                    ]
                                ]
                        ],
                     
                        ['looks_thinkforsecs', 
                            [
                                ['SECS', '2'],
                                ['MESSAGE', 'Hmm...']
                            ]
                        ]]
                    ]
        
        all_blocks_val = self.scratch_parser_inst.get_all_blocks_vals(self.prog1)
        next_val = self.scratch_parser_inst.create_next_values2(all_blocks_val)
        parsed = self.scratch_parser_inst.create_top_tree2(all_blocks_val,next_val)
        
        self.assertEqual(expected2,parsed,msg="Test failed")
             

    def test_scratch_two_arg(self):
        expected_tree1 = ['event_whenflagclicked',[
['motion_movesteps', [
	['STEPS', '10']
		     ]
], 
['motion_turnright', [
	['DEGREES', '15']
		     ]
], 
['looks_sayforsecs', [
	['SECS', '2'], ['MESSAGE', 'Hello!']
		     ]
]
			]
]
        expected_tree2 = ["event_whenflagclicked", [
        ["looks_thinkforsecs", [
            ["SECS", "2"],
            ["MESSAGE", "Hmm..."]
        ]],
        ["motion_pointindirection", [
            ["DIRECTION", "90"]
        ]],
        ["motion_changexby", []
            ["DX", "10"]
        ]
    ]
]
        

        all_blocks_val1 = self.scratch_parser_inst.get_all_blocks_vals(self.prog2)
        all_blocks_val2 = self.scratch_parser_inst.get_all_blocks_vals(self.prog3)
        next_val1 = self.scratch_parser_inst.create_next_values2(all_blocks_val1)
        next_val2 = self.scratch_parser_inst.create_next_values2(all_blocks_val2)
        parsed1 = self.scratch_parser_inst.create_top_tree2(all_blocks_val1,next_val1)
        parsed2 = self.scratch_parser_inst.create_top_tree2(all_blocks_val2,next_val2)

        self.assertEqual(expected_tree1,parsed1,msg="Test failed")
        self.assertEqual(expected_tree2,parsed2,msg="Test failed")


    def test_infinite_loop(self):
        expected = ['event_whenflagclicked', [
		['looks_sayforsecs', [
			['SECS', '2'], ['MESSAGE', 'Infinite with two opcodes in the body']
				     ]
		],
		['control_repeat', [
			'SUBSTACK', [
				'looks_say', [
					[['MESSAGE', 'opcode1']]
					    ], 
				'looks_think', [
					[['MESSAGE', 'opcode2']]
						]
				    ],
			 ['TIMES', '10']
				   ]
		]
			]
]
        all_blocks_val = self.scratch_parser_inst.get_all_blocks_vals(self.prog4)
        next_val = self.scratch_parser_inst.create_next_values2(all_blocks_val)
        parsed = self.scratch_parser_inst.create_top_tree2(all_blocks_val,next_val)
        self.assertEqual(expected,parsed,msg="Test failed")
    def tearDown(self) :
        return self.shortDescription()
        
       
def scratch_suite():
    
    suite = unittest.TestSuite()
    suite.addTest(ScratchTester('test_scratch_nothing_onclick'))
    suite.addTest(ScratchTester('test_scratch_two_arg'))
    suite.addTest(ScratchTester('test_infinite_loop'))
    return suite
    
if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(scratch_suite())

