(* ::Package:: *)

BeginPackage["Simba`"];


JMatrix::usage="Create diagonal J matrix";
SymplecticJ::usage="Check if given matrix is symplectic under J";
ReorderToPairedForm::usage="Reorder state space to paired operator form";
TransformationToRealisable::usage="Find transformation to physically realisable state space";
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

TransformationToRealisable[ss_StateSpaceModel,T_List]:=
Module[{n=SystemsModelOrder@ss,ni,no,j,ji,a,b,c,d,X,vecs,vals},
Clear[x];

{ni,no}=SystemsModelDimensions@ss;
j=JMatrix[n/2];
ji=JMatrix[ni/2];
{a,b,c,d}=Normal[ss];
X=Array[x,{n,n}];

X=X/.Solve[Simplify[a.X+X.a\[ConjugateTranspose]+b.ji.b\[ConjugateTranspose]==ConstantArray[0,{n,n}]]
&& Simplify[X.c\[ConjugateTranspose]+b.ji.d\[ConjugateTranspose]==ConstantArray[0,{n,no}]],Flatten[X]];
Assert[HermitianMatrixQ[X]];

X=X[[1]];
Reduce[Flatten[X - T.j.T\[Transpose]] == 0 && Det[T] != 0, Flatten[T], Reals]];

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
