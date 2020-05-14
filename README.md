# deflated-continuation
Run the deflated continuation code to generate three disconnected bifurcation plots.


# Choosing the right objective
If we already know that the base structure has multiple solutions and we are satisfied we can straigh up maximize the
mileage. 

However, in the case where we are trying to nucleate new stable branches we need to think about another objective. 
Why? In order for additional branches to exist either you go through a point of instability or you find islands ( is
this true). Our algorithm will struggle to find these islands. There is no good algorithm to find islands. 
