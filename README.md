# QASystem
this is a QASystem implemented with BERT

# install prerequisite packages

install with command

```python
pip3 install -U tf-nightly-2.0-preview bert-for-tf2
```

# collect question and answer pairs

put the questions and answers in format as question_answer.txt's. and execute following command to convert the collected samples into dataset format.

```bash
g++ convert.cpp -lboost_filesystem -lboost_system -lboost_program_options -lboost_regex -o convert
./convert -i question_answer.txt -o dataset
```


