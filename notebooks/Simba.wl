(* ::Package:: *)

BeginPackage["Simba`"];


JMatrix::usage="Create diagonal J matrix";
SymplecticJ::usage="Check if given matrix is symplectic under J";
ReorderToPairedForm::usage="Reorder state space to paired operator form";
TransformationToRealisable::usage="Find transformation to physically realisable state space";
TransformationToRealisable::solh="Expected X to be Hermitian";
TransformationToRealisable::solx="Cannot find solution for X";
RealisableQ::usage="Test if StateSpace is physically realisable";


Begin["`Private`"];


JMatrix[n_?Positive]:=DiagonalMatrix[ConstantArray[{1,-1},n]//Flatten];
SymplecticJ[tfmatrix_List, var_Symbol]:=
Module[{jmat=JMatrix[Dimensions[tfmatrix][[1]]/2]},
ConjugateTranspose[tfmatrix/.var->Conjugate[var]].jmat.(tfmatrix/.var->-var)];

(* Swaps rows e.g. (1, 2, 3, 11, 22, 33)^T \[Rule] (1, 11, 2, 22, 3, 33)^T *)
constructPermutationMatrix[n_?Positive]:=
Module[{m=ConstantArray[0,{2 n,2n}]},
Do[m[[2 i-1,i]]=1;m[[2 i,i+n]]=1,{i,n}];
m];
(* Reorders state-space from (a1, a2, a1\[ConjugateTranspose], a2\[ConjugateTranspose]) \[Rule] (a1, a1\[ConjugateTranspose], a2, a2\[ConjugateTranspose]) *)
ReorderToPairedForm[ss_StateSpaceModel]:=
Module[{a,b,c,d},
{a,b,c,d}=Normal[ss];
Module[{u=constructPermutationMatrix[SystemsModelOrder[ss]/2],
ui=constructPermutationMatrix[SystemsModelDimensions[ss][[1]]/2],
uo=constructPermutationMatrix[SystemsModelDimensions[ss][[2]]/2]},
StateSpaceModel[{u.a.Inverse[u],u.b.Inverse[ui],uo.c.Inverse[u],uo.d.Inverse[ui]}]]];

TransformationToRealisable[ss_StateSpaceModel]:=
Module[{n=SystemsModelOrder@ss,ni,no,j,ji,a,b,c,d,X,vecs,vals,sols,eqn},
Clear[x];

{ni,no}=SystemsModelDimensions@ss;
j=JMatrix[n/2];
ji=JMatrix[ni/2];
{a,b,c,d}=Normal[ss];

If[Det[c]==0//Simplify,Message[TransformationToRealisable::solx];Return[]];

(*X=Simplify[(a.b.ji.d\[ConjugateTranspose]-b.ji.b\[ConjugateTranspose].c\[ConjugateTranspose]).Inverse[a\[ConjugateTranspose].c\[ConjugateTranspose]]];*)
X=Simplify[-b.ji.d\[ConjugateTranspose].Inverse[c\[ConjugateTranspose]]];

If[Not[Simplify[X==ConjugateTranspose[X]]],
Message[TransformationToRealisable::solh];Return[];];

(*https://mathematica.stackexchange.com/a/225026/53054*)
Module[{mult,invmult,h2,lu,perm,cnum,diag,lower,newdiag,sqrroots,tmatrix,
wrongpos,wrongmin},
(* From stackexchange: *)
mult = DiagonalMatrix[Table[100^j, {j, 0, n - 1}]];
invmult = Inverse[mult];
h2 = mult.X.mult;
{lu, perm, cnum} = LUDecomposition[h2];
diag = DiagonalMatrix[Diagonal[lu]];
lower = LowerTriangularize[lu] - diag + IdentityMatrix[n];
newdiag = Sign[diag];
sqrroots = Sqrt[Abs[diag]];
tmatrix = invmult.lower.sqrroots;
newdiag=newdiag//Diagonal//Simplify;
(* Joe: Reorder to be in canonical form (1, -1, ...) *)
wrongpos=Map[#[[2]]&,
Select[MapIndexed[{#1==1,First@#2}&,newdiag],First[#]&&EvenQ[#[[2]]]&]];
wrongmin=Map[#[[2]]&,
Select[MapIndexed[{#1==-1,First@#2}&,newdiag],First[#]&&OddQ[#[[2]]]&]];
MapThread[(tmatrix[[All,{#1,#2}]]=tmatrix[[All,{#2,#1}]])&,{wrongpos,wrongmin}];
tmatrix//Simplify]];

RealisableQ[ss_StateSpaceModel]:=
Module[{n=SystemsModelOrder@ss,ni,no,j,ji,a,b,c,d},
{ni,no}=SystemsModelDimensions@ss;
j=JMatrix[n/2];
ji=JMatrix[ni/2];
{a,b,c,d}=Normal[ss];
Simplify[a.j+j.a\[ConjugateTranspose]+b.ji.b\[ConjugateTranspose]==ConstantArray[0,{n,n}]]
&& Simplify[j.c\[ConjugateTranspose]+b.ji.d\[ConjugateTranspose]==ConstantArray[0,{n,no}]]];


End[];


EndPackage[];
