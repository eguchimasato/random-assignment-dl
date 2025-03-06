# random-assignment-dl
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js">
</script>
<script type="text/x-mathjax-config">
 MathJax.Hub.Config({
 tex2jax: {
 inlineMath: [['$', '$'] ],
 displayMath: [ ['$$','$$'], ["\\[","\\]"] ]
 }
 });
</script>


## Impossibility Theorem (Bogomolnaia and Moulin 2001, Theorem 2)
Assume $n\geq 4$. Then there exists no mechanism meeting the three following requirements; ordinal efficiency, strategyproofness, and equal treatment of equals. 

### Ordinal Efficiency
A random assignment $P\in\mathscr{R}$ is ordinally efficient if we have:
```math
\nexists Q\neq P,\;Q_i\;sd(\succ_i)\;P_i\;\text{for all}\;i
```

### Strategyprrofness
A mechanism $P(\cdot)$, namely a mapping from $\mathscr{A}^N$ into $\mathscr{R}$, is strategyproof if we have:
```math
P_i(\succ)\;sd(\succ_i)\;P_i(\succ|^i\succ_i^*)\;\text{for all}\;i\in N,\;\succ_i^*\in\mathscr{A},\;\succ\in\mathscr{A}^N
```

### Equal Treatment of Equals
```math
\succ_i=\succ_j\;\Rightarrow P_i=P_j
```
*Relationship with anonymity

#### Stochastic Dominance
```math
\forall\;P_i,\;Q_i\in\mathscr{L}(A)\;:\;P_i\;sd(\succ_i)\;Q_i\overset{\mathrm{def}}{\iff}\left\{\sum_{k=1}^{t}p_{ia_k}\geq\sum_{k=1}^{t}q_{ia_k},\;\text{for}\;t=1,\dots,n\right\}
```

### Ex-post Efficiency
1. Decompose a biostochastic matrix using Birkhoff algorithm. 
2. Calculate the efficiency violation of each permutatioin matrix. 
3. Calculate the weighted average of the violation. 

cf. [Birkhoff Algorithm](https://en.wikipedia.org/wiki/Birkhoff_algorithm)
