lowerControlEnv.py is a file to test the lower level control.
The goal to carry out is **on(_ingredient_, sink)**.
It has the following options to try in the environment:
1. **--robot** 
    * _default_ 'ur5' 
    * _options_ 'ur5' or 'drake'
2. **--scale** 
    * _default_ 1 
3. **--trials** 
    * _default_ 1 
    * Number of executions to test for the same goal.
4. **--ing_num** 
    * _default_ 0 
    * _options_ 0->pear 1->orange 2->banana 3->tomato.
    * Ingredient to pick and grasp
5. **--test** 
    * _default_ 'separate' 
    * _options_ 'separe' or 'group'
    * Separate tests only the ingredient chosen in _ing_num_. Group tests pick and grasp tasks for all the ingredients the number of trials specified in _trials_.
6. **--random** _default_ False
    * Determine if the positions of the ingredients  are random or fixed.