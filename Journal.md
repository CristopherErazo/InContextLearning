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

# 22-08-2026

I finally managed to understand how the computations of the expectation of the logits transforms into the effective loss for the model and the main difference to the naive approach I was doing initially is the fact that the logits will depend on the position in the sequence and therefore the average over positions to get the full loss is not the same as the loss evaluated on the average logits. 
Once having this, the last thing I focused on was in tracking the matrices $M, Q, \Gamma$ over training and extracting their 'signal' component with which I then used to get the expected logits assuming all the off target positions of the matrices are zero. 
With this setup we get a nice clean expression for the loss but I already know is not correct for several reasons:

* If we track the metrics during training and compute the effective loss in terms of these metrics, it will trivially follow the empirical calculation because you just need the `P_ind` to be computed properly and the logits to increase at a high enough value and the effective loss will sharply decrease from $\log V$ to $\log V(1-P_{ind})$. If the transition is sharp enough, this will be true even if you overestimate the magnitude of the logits. 
* In some experiments, particularly the ones where $d<V$ or large $L$ (I think), the transition sometimes is not sharp and therefore the effective loss fails to closely follow the empirical line near the transition. 
* In some experiment (the ones with small $d$ in particular) I even find that the empirical loss does not decrease all the way to $\log V(1-P_{ind})$, instead it saturates in a higher value (smaller than $\log V$). Whereas on the other hand, the effective loss consistently undershoots the estimation and converges to $\log V(1-P_{ind})$. In this scenario if we observe to the matrices $M, Q, \Gamma$ they all acquire the desired shape.

The previous observation could be due to two reasons:
* My computation of the effective logits is missing some constants (it's not hard to double check but I genuinely doubt it).
* Keeping just the `on` elements of the matrices and setting the rest to zero is not gonna work in specific configurations where the 'small' but 'many' noisy terms can actually compete with the 'signal' which is something I don't take into account so far. 

There are several things I want to do next:
* Squeeze the current model and loss as much as possible to see what can be extracted from it. Hopefully when writing properly the set of ODEs we can estimate the transition time even if the description does not consider the noisy terms yet.
* Parametrice the matrices with more parameters than just the 'diagonal' to write a more complete description of the logits and see if there is any kind of tradeoff between the signal and noise terms.
* So far, tracking just the signal terms there is a parameter $\lambda$ that appears already, but it just depends on $L,V$. The idea would be that if we write the matrices as the optimal superposition of the quenched parameters, we can analyze how the loss depends on the tradeoff between $d,V$ and complete somehow a phase space where we can map the behavior of the model. 
* Ideally I would like to show that the planted model is indeed the solution and maybe that can be done by perturbing the matrices around the planted solution, but it seems a bit hard to compute.
* What is really necessary is to find the way to include the dimensionality into some constants that characterize the model in the same way that $\lambda $ appeared.