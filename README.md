# Sentiment-Analysis
 	Sentiment analysis oh hackerearth predict happiness challenge 
 accuracy - 88.592% (rank 71st of 10,005 participants) pretty bad..isn't it? I think using generalised approach like stacking 
 to assign weights while doing a voting ensemble would have increased accuracy for sure. Another improvement which I 
 think will boost accuracy to upper 90s is using the ultimate text weapon - 'Sent2Vec' which is an extension of 
 'Word2Vec' to create features as it captures syntactic as well as semantics.
 
 	Real Time Twitter Sentiment Analysis
  So I pickled the models created by hackerearth competition and used it for making predictions in another cool thing - 
  Twitter Sentiment Analysis but so as to make it look professional & meaningful I made it end to end. 

	TWITTER ---tweets--stream----> PYTHON MACHINE LEARNING CODE---sentiment--tagged--tweets---> mySQL DB
 Hence as many times I will run it will keep accumulating results and simulataneously store in database can serve for future
 modelling as well *(I can manually tag the incorrect predictions and make model stronger)* . It can stream live tweets thanks to awesome 
 twitter api and predict sentiment of it! I had pretty good results on #flipkart #bigbillionday sale vs #amazon #greatindainsale which 
 analysed twitter sentiments of customers.
      
    
