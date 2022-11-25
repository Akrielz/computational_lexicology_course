import nltk

def nltk_parsing():
    grammar = """
        S -> NP VP
        NP -> NN
        VP -> VBD NP
        NP -> Det NP
        NP -> NN PP
        NP -> DET NN
        PP -> P NP
        NN -> 'I' | 'rainbow' | 'lake'
        VBD -> 'draw'
        Det -> 'the'
        P -> 'on'

        S -> S PP

        PP -> P NN
        NN -> 'Anna'
        P -> 'and'
        NN -> 'Peter'
        VBD -> 'draw'
        NP -> 'us'

        S -> S NP
        VP -> VBD

        NP -> NNS PP
        NP -> 'We'
        VBD -> 'ate'
        NN -> 'octopus'
        P -> 'and'
        NNS -> 'shells'
        P -> 'for'
        NN -> 'dinner'

        PP -> P NNS
        """

    nltk_grammar = nltk.CFG.fromstring(grammar)
    parser = nltk.ChartParser(nltk_grammar)

    # sentence = "I draw the rainbow on the lake".split()
    # for tree in parser.parse(sentence):
    #     print(tree)

    # sentence = "Anna and Peter draw us".split()
    # for tree in parser.parse(sentence):
    #     print(tree)

    sentence = "We ate octopus and shells for dinner".split()
    for tree in parser.parse(sentence):
        print(tree)


if __name__ == "__main__":
    curl_command = "curl http://127.0.0.1:5000/operations > operations.json"
    import os
    os.system(curl_command)
