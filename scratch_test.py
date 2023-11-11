import unittest
from unzip_scratch import unzip_scratch
import json
import sys
from scratch_parser import scratch_parser

class ScratchTester(unittest.TestCase):
        
    
    def setUp(self):
        
        self.scratch_unzipped = unzip_scratch()
        self.prog1 = json.loads(self.scratch_unzipped.unpack_sb3("files/test.sb3"))
        self.prog2 = json.loads(self.scratch_unzipped.unpack_sb3("files/stand_check.sb3"))
        self.scratch_parser_inst = scratch_parser() 
        self.maxDiff = None
        return self.shortDescription()

    def test_scratch_nothing(self):
        expected = {
  "event_whenflagclicked": {
    "looks_sayforsecs": {
      "SECS": "2",
      "MESSAGE": "Hello!"
    },
    "motion_movesteps": {
      "STEPS": "10"
    },
    "control_repeat_until": {
      "CONDITION": {
        "opearator_equals": {
          "OPERAND1": {
            "sensing_answer": {}
          },
          "OPERAND2": "50"
        },
        "sensing_askandwait": {
          "QUESTION": "How old are you?"
        }
      }
    },
    "looks_thinkforsecs": {
      "SECS": "2",
      "MESSAGE": "Hmm..."
    }
  }
}
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
        
   
    def test_scratch_nothing2(self):
        expected = {
    "event_whenflagclicked": {
        "motion_movesteps": {
            "STEPS": "10"
        },
        "motion_turnright": {
            "DEGREES": "15"
        },
        "looks_sayforsecs": {
            "SECS": "2",
            "MESSAGE": "Hello!"
        }
    }
}
        
        all_blocks_val = self.scratch_parser_inst.get_all_blocks_vals(self.prog2)
        next_val = self.scratch_parser_inst.create_next_values(all_blocks_val)
        parsed = self.scratch_parser_inst.create_top_tree(all_blocks_val,next_val)

        self.assertEqual(expected,parsed,msg="Test failed")

    def test_scratch_two_arg(self):
        expected = {}


    def tearDown(self) :
        return self.shortDescription()
        
       
def scratch_suite():
    
    suite = unittest.TestSuite()
    suite.addTest(ScratchTester('test_scratch_nothing'))
    suite.addTest(ScratchTester('test_scratch_nothing2'))
    return suite
    
if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(scratch_suite())

