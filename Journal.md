# 14-08-2026

I discovered that the thing that might me messing things up in the sense that I don't get the same magnitud of `h_star` even if the general shape correctly respects what is observed is because in the `on_logit_mean` variable I am computing the average of the logits over all possible positions in the different batches that admit the posibility of induction i.e $\{ \tau_\mu \in \mathcal{T}\; and \; \exist \nu<\mu: \tau_\nu=\tau_\mu\}$. Nonetheless, for the theoretical calculation first I compute the expectation of the logits at a given position $\mu$ conditioned that that position admits induction and then I'm trying to match `on_logit_mean` by taking a simple average over $\mu = 1,\cdots,L$.

I think what I should do is one of two things (or the two):

* Refine the average over $\mu$ in the theoretical computation such that we include the fact that not all positions admit for induction. For this I would need to write a formula for $P(\mu)$ which is the probability that position $\mu$ admits induction. This migth be a bit hard in practice but it can even be computed exactly numerically. 
* Extract better the logits I compute `on_logit_mean` from to keep track also of the $\mu$ values where the logit comes from such that I can perform a first average `on_logit_mean_per_position (L,)` and then `on_logit_mean = on_logit_mean_per_position.mean()` but I would have a more detailed information to contrast my theory with.

Merging these things I believe it can also be explained the bimodality of the on-target distribution in the sense that if we increase $\mu$ we should see the bimodality increasing (or the second peak appearing).

I also changed the scales of the matrices at initialization and the normalization of the attention layer and test its scaling and so far is good and works. What I cannot explain yet is the change of scale of the order parameters as I increase $V$. It seems that they are not $O(1)$ yet but I don't know what is the right normalization. So far I normalized $M,Q$ by $1/\sqrt(d)$ because it's the term it popped out when I changed the values of the normalization constant in the attention layers and I thought that was enough but instead it isn't and the parameters seem to be still scaling with $V$ in some way. 

A disadvantage of having $V,d,L,K$ coupled now with their respective $\alpha_* = */V$ or similar is that I cannot isolate what variable is the one causing the order parameters to scale. Maybe I would need to go back and decouple them for now to fix the experiments first. 

# 15-08-2026

I found a way to divide the output logits in the places where induction is possible such that now we can compare both the theoretical prediction and the aggregated loss if that is the case.

I tested the numerical estimate of the induction probability at position $\mu$ and it coincides quite nicely with the theoretical formula $P(\mu) = \frac{\rho}{1+\rho} \times (1- \exp{(\frac{-\mu}{(1+\rho)V})}$

I obtained the variances of the 'un-normalized' order parameters (simply the sums) and now I want to test if they respect scaling as dimensions vary.