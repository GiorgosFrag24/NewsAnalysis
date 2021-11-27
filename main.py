import helper_functions as h
from helper_functions import generate_bigrams
import sys
import argparse
exec(open("spacy_vocab_download.py").read())

parser = argparse.ArgumentParser(description='Sentiment analysis program using the NewsAPI')

parser.add_argument('search', help='A required string keyword search argument')

parser.add_argument('--sentiment', action="store_true" ,
                    help='An optional sentiment analysis argument, defaults to Naive Bayes approach')

parser.add_argument('--method',
                    help='''An optional method selector switch, A = Naive Bayes\n
                    B = Pattern Analyzer, C = Custom Trained Model ''')
parser.add_argument('--year', type=int,
                    help='An optional timeframe selection (currently not working due to plan limitations)')
args = parser.parse_args()
if args.year!=None:
    response = h.apiRequest(args.search,year=args.year)
else:    
    response = h.apiRequest(args.search)

if args.sentiment==False:
     h.listSources(response)    
elif (args.sentiment==True): 
     analyzer = h.SentimentAnalyzer(response)
     if args.method=='B':
         analyzer.PatternAnalyzer()
     elif args.method=='C':   
         analyzer.Custom()
     else:
         analyzer.BayesAnalyzer()
 
        
