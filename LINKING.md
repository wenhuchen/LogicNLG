This README page is aimed to explain the rule-based parser and entity linker we used in the system

- Entity Linking System:
  You can test with the entity-linking system with following command
  ```
  python parse_programs.py --preprocess
  ```
  This command will output the linked sentence and their corresponding columns. The procedure is described as follows:
  ```
    sent, pos_tags = self.normalize(sent)
    raw_sent = " ".join(sent)
    linked_sent, pos = self.entity_link(table_name, sent, pos_tags)
    masked_sent, mem_str, mem_num, head_str, head_num, mapping = self.initialize_buffer(table_name, linked_sent, pos, raw_sent)
  ```
  We first normalize the sentence to get its tokenized form and the POS-TAG, and then we run greedy search to find out the longest string match of each n-gram chunk of the sentence in the table using self.entity_link function. Finally, we will distinguish the types of the linked entities. Specifically, there are three types, namely <ENTITY>, <COUNT> and <COMPUTE>.
  1. <ENTITY> contains the names, location, numbers, etc directly seen in the table.
  2. <COUNT> contains the counting numbers like "2, 3, 4, ..." not directly seen in the table.
  3. <COMPUTE> contains the numeric values which are neither count nor seen in the table, like average, difference, comparison number, etc.
  These fine-grained three types of slots are used in the following stage. For example, when generating adversarial examples, you can replace the <entity> or <count> or <compute> with some confusing entities from nowhere.
  When generating our train/val/test_lm.py, we use a coarse-grained linking to replace all these three types of entities with simply [ENT], which turns out to perform slightly better than using the fine-grained version.

- Program Searching:
  You can run the following command to generate the parses for the training data, which can be further used for the program ranker model (BERT-style program/text matching).
  ```
  python parse_programs.py --gen_prog
  python parse_programs.py --rank_prog_bert
  ```
  The searching procedure is mainly through breath-first search using the following code, which takes as input the initialized buffer from the previous entity linking system
  ```
  result = self.run(table_name, raw_sent, masked_sent, pos, mem_str, mem_num, head_str, head_num)
  ```
  The maximum steps of search is constrained to 7 steps, and the timeout is set to 50 seconds. The searched programs need to consume all the buffer values and return True/False in the final step.
  
  
