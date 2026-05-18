# 计算复杂性：P

<div style="text-align:center">
    王艺杭<br>
    2023202316
</div>
1. 设 $\text{CNF}_k = \{\langle\phi\rangle \mid \phi \text{ 是可满足的 CNF 公式，且每个变量最多出现 } k \text{ 次}\}$

   1. 证明 $\text{CNF}_2 \in P$。

      设 $T_p$ 为 $\text{CNF}_2$ 的多项式时间判定机，描述如下：

      > $T_p$ = "对输入 $\langle\phi\rangle$："

      **算法步骤：**

      1. 考察 $\phi$ 的第一个子句。若其形如 $x$，且 $\phi$ 中存在子句 $\neg x$，则**拒绝**。
      2. 若 CNF 形如 $x \lor A$（其中 $A$ 是 CNF）：若 $x$ 在其他子句中不以否定形式出现，则从 $\phi$ 中删除所有形如 $x \lor B$ 的子句，得到 $\phi'$；若 $\phi'$ 中无子句，则**接受**。
      3. 对 $c$ 仅以正文字出现（无否定）的情况，直接求解。
      4. 搜索 $\phi$ 时，若找到形如 $x \lor A$ 和 $\neg x \lor B$ 的子句，则将其删除，并将 $A \lor B$ 加入 $\phi$。
      5. 返回步骤 1。

      **复杂度分析：**

      $T_p$ 每次处理变量后，$\phi$ 中的子句数减少 1 或 2，因此运行时间关于变量数是**多项式时间**的。

      $$\boxed{\text{CNF}_2 \in P}$$

   2. 证明 $\text{CNF}_3$ 是 NP 完全的。

      NP 完全需满足两个条件：

      1. $B \in \text{NP}$
      2. $\text{NP}$ 中每个问题都可多项式时间归约到 $B$

      构造多项式时间验证机 $V_p$：

      > $V_p$ = "对输入 $\langle\langle\phi\rangle, x\rangle$："

      - 验证 $\phi$ 中每个变量最多出现 3 次；
      - 验证 $x$ 是 $\phi$ 的一个满足赋值；
      - 若两个条件均满足，则**接受**；否则**拒绝**。

      设 $r_p$ 为从 3SAT 到 $\text{CNF}_3$ 的多项式时间归约。

      对于 3SAT 的输入实例 $\phi$，构造 $\text{CNF}_3$ 的实例 $r_p(\langle\phi\rangle)$：

      1. 从左到右扫描，找到在公式中出现超过 3 次的变量。设变量 $S$ 出现在 $m$ 个位置：

      $$
      (x_1 \lor A_1), \ldots, (x_m \lor A_m) \quad \text{其中 } x_i \text{ 为 } S \text{ 或 } \neg S
      $$

      2. 若无变量出现超过 3 次，输出 $\phi$。
      3. 引入新变量 $S_1, \ldots, S_m$，对任意 $(x_i \lor A_i)$，从公式中删除并替换。
      4. 添加以下子句链（"一致性约束"）：

      $$
      (S_1 \lor A_1) \land (\neg S_1 \lor S_2) \land (S_2 \lor A_2) \land (\neg S_2 \lor S_3) \land \cdots \land (S_m \lor A_m) \land (\neg S_m \lor S_m)
      $$

      5. 返回步骤 1。

      **正确性：**

      - $r_p(\langle\phi\rangle)$ 保证每个变量最多出现 3 次；
      - $\phi$ 可满足当且仅当 $r_p(\langle\phi\rangle)$ 可满足；
      - $r_p$ 关于 $\phi$ 中变量数是多项式时间的。

      由条件 (1) 和 (2)：

      $$\boxed{\text{CNF}_3 \text{ 是 NP 完全的}}$$



2. 设 $\text{CNF}_H = \{\langle\phi\rangle \mid \phi \text{ 是可满足的 CNF 公式，且每个子句包含任意数量的文字，但最多只有一个否定文字 }\}$. **证明 $\text{CNF}_H \in P$。**

   构造判定机 $N$：

   > $N$ = "对输入 $\langle\phi\rangle$，其中 $\phi$ 是一个 CNF 布尔公式："

   **算法步骤：**

   1. 若 $\phi$ 中**不包含**形如 $(\neg x)$ 的单元子句，则令所有正文字 $x = 1$，所有否定文字 $\neg x = 0$，**接受**。
   2. **重复执行**，直到不再出现新的 $(\neg x)$ 单元子句：
   3. 若 $\phi$ 包含单元子句 $(\neg x)$：
      - 从 $\phi$ 中删除所有**包含** $\neg x$ 的子句；
      - 从 $\phi$ 中删除所有子句中**出现的** $x$（即正文字 $x$）。
   4. 若 $\phi$ 中出现**空子句**，则**拒绝**。
   5. 令 $\phi$ 中所有剩余正文字 $x = 1$，否定文字 $\neg x = 0$，**接受**。

   考察 $\phi$ 的第一个子句：

   - 若其属于某变量 $x$，且存在 $\neg x$，则**拒绝**；
   - 若属于另一变量 $y$，且不存在 $\neg y$，则**接受**。

   从 $\phi$ 中移除这两种情形，将其结果（记为 $M$ 和 $N$）合并回 $\phi$，即可证明 $\text{CNF}_2 \in P$。

   考察 $\phi$ 的第一个子句：

   - 若其属于某变量 $p$，且存在 $\neg p$，则**拒绝**；
   - 若属于另一变量 $q$，且不存在 $\neg q$，则**接受**；
   - 若涉及变量 $r$，且不存在 $\neg q$，同样**接受**。

   从 $\phi$ 中移除上述情形，将其结果（记为 $A$ 和 $B$）合并回 $\phi$，即可证明 $\text{CNF}_3 \in P$。

   由于 $\text{CNF}_H$ 中每个子句**至多含一个否定文字**，上述单元传播（Unit Propagation）算法可在**多项式时间**内完成判定。将 $\text{CNF}_2$ 和 $\text{CNF}_3$ 的结论推广，对任意满足该结构约束的 CNF 公式均成立：

   $$\boxed{\text{CNF}_H \in P}$$



