Лю
ѕФ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58вш
Ј
(Adam/v/deep_q_network2d_16/dense_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/v/deep_q_network2d_16/dense_50/bias
Ё
<Adam/v/deep_q_network2d_16/dense_50/bias/Read/ReadVariableOpReadVariableOp(Adam/v/deep_q_network2d_16/dense_50/bias*
_output_shapes
:*
dtype0
Ј
(Adam/m/deep_q_network2d_16/dense_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/m/deep_q_network2d_16/dense_50/bias
Ё
<Adam/m/deep_q_network2d_16/dense_50/bias/Read/ReadVariableOpReadVariableOp(Adam/m/deep_q_network2d_16/dense_50/bias*
_output_shapes
:*
dtype0
А
*Adam/v/deep_q_network2d_16/dense_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*;
shared_name,*Adam/v/deep_q_network2d_16/dense_50/kernel
Љ
>Adam/v/deep_q_network2d_16/dense_50/kernel/Read/ReadVariableOpReadVariableOp*Adam/v/deep_q_network2d_16/dense_50/kernel*
_output_shapes

:@*
dtype0
А
*Adam/m/deep_q_network2d_16/dense_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*;
shared_name,*Adam/m/deep_q_network2d_16/dense_50/kernel
Љ
>Adam/m/deep_q_network2d_16/dense_50/kernel/Read/ReadVariableOpReadVariableOp*Adam/m/deep_q_network2d_16/dense_50/kernel*
_output_shapes

:@*
dtype0
Ј
(Adam/v/deep_q_network2d_16/dense_49/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(Adam/v/deep_q_network2d_16/dense_49/bias
Ё
<Adam/v/deep_q_network2d_16/dense_49/bias/Read/ReadVariableOpReadVariableOp(Adam/v/deep_q_network2d_16/dense_49/bias*
_output_shapes
:@*
dtype0
Ј
(Adam/m/deep_q_network2d_16/dense_49/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(Adam/m/deep_q_network2d_16/dense_49/bias
Ё
<Adam/m/deep_q_network2d_16/dense_49/bias/Read/ReadVariableOpReadVariableOp(Adam/m/deep_q_network2d_16/dense_49/bias*
_output_shapes
:@*
dtype0
А
*Adam/v/deep_q_network2d_16/dense_49/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*;
shared_name,*Adam/v/deep_q_network2d_16/dense_49/kernel
Љ
>Adam/v/deep_q_network2d_16/dense_49/kernel/Read/ReadVariableOpReadVariableOp*Adam/v/deep_q_network2d_16/dense_49/kernel*
_output_shapes

:@@*
dtype0
А
*Adam/m/deep_q_network2d_16/dense_49/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*;
shared_name,*Adam/m/deep_q_network2d_16/dense_49/kernel
Љ
>Adam/m/deep_q_network2d_16/dense_49/kernel/Read/ReadVariableOpReadVariableOp*Adam/m/deep_q_network2d_16/dense_49/kernel*
_output_shapes

:@@*
dtype0
Ј
(Adam/v/deep_q_network2d_16/dense_48/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(Adam/v/deep_q_network2d_16/dense_48/bias
Ё
<Adam/v/deep_q_network2d_16/dense_48/bias/Read/ReadVariableOpReadVariableOp(Adam/v/deep_q_network2d_16/dense_48/bias*
_output_shapes
:@*
dtype0
Ј
(Adam/m/deep_q_network2d_16/dense_48/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(Adam/m/deep_q_network2d_16/dense_48/bias
Ё
<Adam/m/deep_q_network2d_16/dense_48/bias/Read/ReadVariableOpReadVariableOp(Adam/m/deep_q_network2d_16/dense_48/bias*
_output_shapes
:@*
dtype0
А
*Adam/v/deep_q_network2d_16/dense_48/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*;
shared_name,*Adam/v/deep_q_network2d_16/dense_48/kernel
Љ
>Adam/v/deep_q_network2d_16/dense_48/kernel/Read/ReadVariableOpReadVariableOp*Adam/v/deep_q_network2d_16/dense_48/kernel*
_output_shapes

:@*
dtype0
А
*Adam/m/deep_q_network2d_16/dense_48/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*;
shared_name,*Adam/m/deep_q_network2d_16/dense_48/kernel
Љ
>Adam/m/deep_q_network2d_16/dense_48/kernel/Read/ReadVariableOpReadVariableOp*Adam/m/deep_q_network2d_16/dense_48/kernel*
_output_shapes

:@*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	

!deep_q_network2d_16/dense_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!deep_q_network2d_16/dense_50/bias

5deep_q_network2d_16/dense_50/bias/Read/ReadVariableOpReadVariableOp!deep_q_network2d_16/dense_50/bias*
_output_shapes
:*
dtype0
Ђ
#deep_q_network2d_16/dense_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*4
shared_name%#deep_q_network2d_16/dense_50/kernel

7deep_q_network2d_16/dense_50/kernel/Read/ReadVariableOpReadVariableOp#deep_q_network2d_16/dense_50/kernel*
_output_shapes

:@*
dtype0

!deep_q_network2d_16/dense_49/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!deep_q_network2d_16/dense_49/bias

5deep_q_network2d_16/dense_49/bias/Read/ReadVariableOpReadVariableOp!deep_q_network2d_16/dense_49/bias*
_output_shapes
:@*
dtype0
Ђ
#deep_q_network2d_16/dense_49/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*4
shared_name%#deep_q_network2d_16/dense_49/kernel

7deep_q_network2d_16/dense_49/kernel/Read/ReadVariableOpReadVariableOp#deep_q_network2d_16/dense_49/kernel*
_output_shapes

:@@*
dtype0

!deep_q_network2d_16/dense_48/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!deep_q_network2d_16/dense_48/bias

5deep_q_network2d_16/dense_48/bias/Read/ReadVariableOpReadVariableOp!deep_q_network2d_16/dense_48/bias*
_output_shapes
:@*
dtype0
Ђ
#deep_q_network2d_16/dense_48/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*4
shared_name%#deep_q_network2d_16/dense_48/kernel

7deep_q_network2d_16/dense_48/kernel/Read/ReadVariableOpReadVariableOp#deep_q_network2d_16/dense_48/kernel*
_output_shapes

:@*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1#deep_q_network2d_16/dense_48/kernel!deep_q_network2d_16/dense_48/bias#deep_q_network2d_16/dense_49/kernel!deep_q_network2d_16/dense_49/bias#deep_q_network2d_16/dense_50/kernel!deep_q_network2d_16/dense_50/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 */
f*R(
&__inference_signature_wrapper_30853296

NoOpNoOp
Х(
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*(
valueі'Bѓ' Bь'
ё
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
fc1
	fc2

fc3
	optimizer
loss

signatures*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
А
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
І
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

kernel
bias*
І
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

kernel
bias*
І
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

kernel
bias*

/
_variables
0_iterations
1_learning_rate
2_index_dict
3
_momentums
4_velocities
5_update_step_xla*
* 

6serving_default* 
c]
VARIABLE_VALUE#deep_q_network2d_16/dense_48/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!deep_q_network2d_16/dense_48/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#deep_q_network2d_16/dense_49/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!deep_q_network2d_16/dense_49/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#deep_q_network2d_16/dense_50/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!deep_q_network2d_16/dense_50/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
	1

2*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 

7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

<trace_0* 

=trace_0* 

0
1*

0
1*
* 

>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

Ctrace_0* 

Dtrace_0* 

0
1*

0
1*
* 

Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

Jtrace_0* 

Ktrace_0* 
b
00
L1
M2
N3
O4
P5
Q6
R7
S8
T9
U10
V11
W12*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
L0
N1
P2
R3
T4
V5*
.
M0
O1
Q2
S3
U4
W5*
P
Xtrace_0
Ytrace_1
Ztrace_2
[trace_3
\trace_4
]trace_5* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
uo
VARIABLE_VALUE*Adam/m/deep_q_network2d_16/dense_48/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/v/deep_q_network2d_16/dense_48/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/m/deep_q_network2d_16/dense_48/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/v/deep_q_network2d_16/dense_48/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/m/deep_q_network2d_16/dense_49/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/v/deep_q_network2d_16/dense_49/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/m/deep_q_network2d_16/dense_49/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/v/deep_q_network2d_16/dense_49/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/m/deep_q_network2d_16/dense_50/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/v/deep_q_network2d_16/dense_50/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/deep_q_network2d_16/dense_50/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/deep_q_network2d_16/dense_50/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
М
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename7deep_q_network2d_16/dense_48/kernel/Read/ReadVariableOp5deep_q_network2d_16/dense_48/bias/Read/ReadVariableOp7deep_q_network2d_16/dense_49/kernel/Read/ReadVariableOp5deep_q_network2d_16/dense_49/bias/Read/ReadVariableOp7deep_q_network2d_16/dense_50/kernel/Read/ReadVariableOp5deep_q_network2d_16/dense_50/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp>Adam/m/deep_q_network2d_16/dense_48/kernel/Read/ReadVariableOp>Adam/v/deep_q_network2d_16/dense_48/kernel/Read/ReadVariableOp<Adam/m/deep_q_network2d_16/dense_48/bias/Read/ReadVariableOp<Adam/v/deep_q_network2d_16/dense_48/bias/Read/ReadVariableOp>Adam/m/deep_q_network2d_16/dense_49/kernel/Read/ReadVariableOp>Adam/v/deep_q_network2d_16/dense_49/kernel/Read/ReadVariableOp<Adam/m/deep_q_network2d_16/dense_49/bias/Read/ReadVariableOp<Adam/v/deep_q_network2d_16/dense_49/bias/Read/ReadVariableOp>Adam/m/deep_q_network2d_16/dense_50/kernel/Read/ReadVariableOp>Adam/v/deep_q_network2d_16/dense_50/kernel/Read/ReadVariableOp<Adam/m/deep_q_network2d_16/dense_50/bias/Read/ReadVariableOp<Adam/v/deep_q_network2d_16/dense_50/bias/Read/ReadVariableOpConst*!
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 **
f%R#
!__inference__traced_save_30853479
Ї
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename#deep_q_network2d_16/dense_48/kernel!deep_q_network2d_16/dense_48/bias#deep_q_network2d_16/dense_49/kernel!deep_q_network2d_16/dense_49/bias#deep_q_network2d_16/dense_50/kernel!deep_q_network2d_16/dense_50/bias	iterationlearning_rate*Adam/m/deep_q_network2d_16/dense_48/kernel*Adam/v/deep_q_network2d_16/dense_48/kernel(Adam/m/deep_q_network2d_16/dense_48/bias(Adam/v/deep_q_network2d_16/dense_48/bias*Adam/m/deep_q_network2d_16/dense_49/kernel*Adam/v/deep_q_network2d_16/dense_49/kernel(Adam/m/deep_q_network2d_16/dense_49/bias(Adam/v/deep_q_network2d_16/dense_49/bias*Adam/m/deep_q_network2d_16/dense_50/kernel*Adam/v/deep_q_network2d_16/dense_50/kernel(Adam/m/deep_q_network2d_16/dense_50/bias(Adam/v/deep_q_network2d_16/dense_50/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *-
f(R&
$__inference__traced_restore_30853549є
Й
P
$__inference__update_step_xla_2493591
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:@: *
	_noinline(:H D

_output_shapes

:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Й
P
$__inference__update_step_xla_2493571
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:@: *
	_noinline(:H D

_output_shapes

:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable


ї
F__inference_dense_48_layer_call_and_return_conditional_losses_30853154

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Щ	
ї
F__inference_dense_50_layer_call_and_return_conditional_losses_30853396

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ы

+__inference_dense_48_layer_call_fn_30853346

inputs
unknown:@
	unknown_0:@
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dense_48_layer_call_and_return_conditional_losses_30853154o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Л\
Ѕ
$__inference__traced_restore_30853549
file_prefixF
4assignvariableop_deep_q_network2d_16_dense_48_kernel:@B
4assignvariableop_1_deep_q_network2d_16_dense_48_bias:@H
6assignvariableop_2_deep_q_network2d_16_dense_49_kernel:@@B
4assignvariableop_3_deep_q_network2d_16_dense_49_bias:@H
6assignvariableop_4_deep_q_network2d_16_dense_50_kernel:@B
4assignvariableop_5_deep_q_network2d_16_dense_50_bias:&
assignvariableop_6_iteration:	 *
 assignvariableop_7_learning_rate: O
=assignvariableop_8_adam_m_deep_q_network2d_16_dense_48_kernel:@O
=assignvariableop_9_adam_v_deep_q_network2d_16_dense_48_kernel:@J
<assignvariableop_10_adam_m_deep_q_network2d_16_dense_48_bias:@J
<assignvariableop_11_adam_v_deep_q_network2d_16_dense_48_bias:@P
>assignvariableop_12_adam_m_deep_q_network2d_16_dense_49_kernel:@@P
>assignvariableop_13_adam_v_deep_q_network2d_16_dense_49_kernel:@@J
<assignvariableop_14_adam_m_deep_q_network2d_16_dense_49_bias:@J
<assignvariableop_15_adam_v_deep_q_network2d_16_dense_49_bias:@P
>assignvariableop_16_adam_m_deep_q_network2d_16_dense_50_kernel:@P
>assignvariableop_17_adam_v_deep_q_network2d_16_dense_50_kernel:@J
<assignvariableop_18_adam_m_deep_q_network2d_16_dense_50_bias:J
<assignvariableop_19_adam_v_deep_q_network2d_16_dense_50_bias:
identity_21ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ы
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ё
valueчBфB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOpAssignVariableOp4assignvariableop_deep_q_network2d_16_dense_48_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_1AssignVariableOp4assignvariableop_1_deep_q_network2d_16_dense_48_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_2AssignVariableOp6assignvariableop_2_deep_q_network2d_16_dense_49_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_3AssignVariableOp4assignvariableop_3_deep_q_network2d_16_dense_49_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_4AssignVariableOp6assignvariableop_4_deep_q_network2d_16_dense_50_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_5AssignVariableOp4assignvariableop_5_deep_q_network2d_16_dense_50_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:Г
AssignVariableOp_6AssignVariableOpassignvariableop_6_iterationIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_8AssignVariableOp=assignvariableop_8_adam_m_deep_q_network2d_16_dense_48_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_9AssignVariableOp=assignvariableop_9_adam_v_deep_q_network2d_16_dense_48_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_10AssignVariableOp<assignvariableop_10_adam_m_deep_q_network2d_16_dense_48_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_11AssignVariableOp<assignvariableop_11_adam_v_deep_q_network2d_16_dense_48_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_12AssignVariableOp>assignvariableop_12_adam_m_deep_q_network2d_16_dense_49_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_13AssignVariableOp>assignvariableop_13_adam_v_deep_q_network2d_16_dense_49_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_14AssignVariableOp<assignvariableop_14_adam_m_deep_q_network2d_16_dense_49_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_15AssignVariableOp<assignvariableop_15_adam_v_deep_q_network2d_16_dense_49_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_16AssignVariableOp>assignvariableop_16_adam_m_deep_q_network2d_16_dense_50_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_17AssignVariableOp>assignvariableop_17_adam_v_deep_q_network2d_16_dense_50_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_18AssignVariableOp<assignvariableop_18_adam_m_deep_q_network2d_16_dense_50_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_19AssignVariableOp<assignvariableop_19_adam_v_deep_q_network2d_16_dense_50_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_21IdentityIdentity_20:output:0^NoOp_1*
T0*
_output_shapes
: є
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_21Identity_21:output:0*=
_input_shapes,
*: : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ю

&__inference_signature_wrapper_30853296
input_1
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *,
f'R%
#__inference__wrapped_model_30853136o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
З
Њ
Q__inference_deep_q_network2d_16_layer_call_and_return_conditional_losses_30853194	
state#
dense_48_30853155:@
dense_48_30853157:@#
dense_49_30853172:@@
dense_49_30853174:@#
dense_50_30853188:@
dense_50_30853190:
identityЂ dense_48/StatefulPartitionedCallЂ dense_49/StatefulPartitionedCallЂ dense_50/StatefulPartitionedCallњ
 dense_48/StatefulPartitionedCallStatefulPartitionedCallstatedense_48_30853155dense_48_30853157*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dense_48_layer_call_and_return_conditional_losses_30853154
 dense_49/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0dense_49_30853172dense_49_30853174*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dense_49_layer_call_and_return_conditional_losses_30853171
 dense_50/StatefulPartitionedCallStatefulPartitionedCall)dense_49/StatefulPartitionedCall:output:0dense_50_30853188dense_50_30853190*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dense_50_layer_call_and_return_conditional_losses_30853187x
IdentityIdentity)dense_50/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЏ
NoOpNoOp!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall:N J
'
_output_shapes
:џџџџџџџџџ

_user_specified_namestate
­
L
$__inference__update_step_xla_2493596
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Щ	
ї
F__inference_dense_50_layer_call_and_return_conditional_losses_30853187

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Х

Q__inference_deep_q_network2d_16_layer_call_and_return_conditional_losses_30853337	
state9
'dense_48_matmul_readvariableop_resource:@6
(dense_48_biasadd_readvariableop_resource:@9
'dense_49_matmul_readvariableop_resource:@@6
(dense_49_biasadd_readvariableop_resource:@9
'dense_50_matmul_readvariableop_resource:@6
(dense_50_biasadd_readvariableop_resource:
identityЂdense_48/BiasAdd/ReadVariableOpЂdense_48/MatMul/ReadVariableOpЂdense_49/BiasAdd/ReadVariableOpЂdense_49/MatMul/ReadVariableOpЂdense_50/BiasAdd/ReadVariableOpЂdense_50/MatMul/ReadVariableOp
dense_48/MatMul/ReadVariableOpReadVariableOp'dense_48_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0z
dense_48/MatMulMatMulstate&dense_48/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_48/BiasAdd/ReadVariableOpReadVariableOp(dense_48_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_48/BiasAddBiasAdddense_48/MatMul:product:0'dense_48/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@b
dense_48/ReluReludense_48/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_49/MatMul/ReadVariableOpReadVariableOp'dense_49_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0
dense_49/MatMulMatMuldense_48/Relu:activations:0&dense_49/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_49/BiasAdd/ReadVariableOpReadVariableOp(dense_49_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_49/BiasAddBiasAdddense_49/MatMul:product:0'dense_49/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@b
dense_49/ReluReludense_49/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_50/MatMul/ReadVariableOpReadVariableOp'dense_50_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_50/MatMulMatMuldense_49/Relu:activations:0&dense_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_50/BiasAdd/ReadVariableOpReadVariableOp(dense_50_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_50/BiasAddBiasAdddense_50/MatMul:product:0'dense_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџh
IdentityIdentitydense_50/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp ^dense_48/BiasAdd/ReadVariableOp^dense_48/MatMul/ReadVariableOp ^dense_49/BiasAdd/ReadVariableOp^dense_49/MatMul/ReadVariableOp ^dense_50/BiasAdd/ReadVariableOp^dense_50/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 2B
dense_48/BiasAdd/ReadVariableOpdense_48/BiasAdd/ReadVariableOp2@
dense_48/MatMul/ReadVariableOpdense_48/MatMul/ReadVariableOp2B
dense_49/BiasAdd/ReadVariableOpdense_49/BiasAdd/ReadVariableOp2@
dense_49/MatMul/ReadVariableOpdense_49/MatMul/ReadVariableOp2B
dense_50/BiasAdd/ReadVariableOpdense_50/BiasAdd/ReadVariableOp2@
dense_50/MatMul/ReadVariableOpdense_50/MatMul/ReadVariableOp:N J
'
_output_shapes
:џџџџџџџџџ

_user_specified_namestate
Н
Ќ
Q__inference_deep_q_network2d_16_layer_call_and_return_conditional_losses_30853275
input_1#
dense_48_30853259:@
dense_48_30853261:@#
dense_49_30853264:@@
dense_49_30853266:@#
dense_50_30853269:@
dense_50_30853271:
identityЂ dense_48/StatefulPartitionedCallЂ dense_49/StatefulPartitionedCallЂ dense_50/StatefulPartitionedCallќ
 dense_48/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_48_30853259dense_48_30853261*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dense_48_layer_call_and_return_conditional_losses_30853154
 dense_49/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0dense_49_30853264dense_49_30853266*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dense_49_layer_call_and_return_conditional_losses_30853171
 dense_50/StatefulPartitionedCallStatefulPartitionedCall)dense_49/StatefulPartitionedCall:output:0dense_50_30853269dense_50_30853271*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dense_50_layer_call_and_return_conditional_losses_30853187x
IdentityIdentity)dense_50/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЏ
NoOpNoOp!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
	

6__inference_deep_q_network2d_16_layer_call_fn_30853209
input_1
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identityЂStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_deep_q_network2d_16_layer_call_and_return_conditional_losses_30853194o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
Ў5
а
!__inference__traced_save_30853479
file_prefixB
>savev2_deep_q_network2d_16_dense_48_kernel_read_readvariableop@
<savev2_deep_q_network2d_16_dense_48_bias_read_readvariableopB
>savev2_deep_q_network2d_16_dense_49_kernel_read_readvariableop@
<savev2_deep_q_network2d_16_dense_49_bias_read_readvariableopB
>savev2_deep_q_network2d_16_dense_50_kernel_read_readvariableop@
<savev2_deep_q_network2d_16_dense_50_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableopI
Esavev2_adam_m_deep_q_network2d_16_dense_48_kernel_read_readvariableopI
Esavev2_adam_v_deep_q_network2d_16_dense_48_kernel_read_readvariableopG
Csavev2_adam_m_deep_q_network2d_16_dense_48_bias_read_readvariableopG
Csavev2_adam_v_deep_q_network2d_16_dense_48_bias_read_readvariableopI
Esavev2_adam_m_deep_q_network2d_16_dense_49_kernel_read_readvariableopI
Esavev2_adam_v_deep_q_network2d_16_dense_49_kernel_read_readvariableopG
Csavev2_adam_m_deep_q_network2d_16_dense_49_bias_read_readvariableopG
Csavev2_adam_v_deep_q_network2d_16_dense_49_bias_read_readvariableopI
Esavev2_adam_m_deep_q_network2d_16_dense_50_kernel_read_readvariableopI
Esavev2_adam_v_deep_q_network2d_16_dense_50_kernel_read_readvariableopG
Csavev2_adam_m_deep_q_network2d_16_dense_50_bias_read_readvariableopG
Csavev2_adam_v_deep_q_network2d_16_dense_50_bias_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ш
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ё
valueчBфB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B ў
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0>savev2_deep_q_network2d_16_dense_48_kernel_read_readvariableop<savev2_deep_q_network2d_16_dense_48_bias_read_readvariableop>savev2_deep_q_network2d_16_dense_49_kernel_read_readvariableop<savev2_deep_q_network2d_16_dense_49_bias_read_readvariableop>savev2_deep_q_network2d_16_dense_50_kernel_read_readvariableop<savev2_deep_q_network2d_16_dense_50_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableopEsavev2_adam_m_deep_q_network2d_16_dense_48_kernel_read_readvariableopEsavev2_adam_v_deep_q_network2d_16_dense_48_kernel_read_readvariableopCsavev2_adam_m_deep_q_network2d_16_dense_48_bias_read_readvariableopCsavev2_adam_v_deep_q_network2d_16_dense_48_bias_read_readvariableopEsavev2_adam_m_deep_q_network2d_16_dense_49_kernel_read_readvariableopEsavev2_adam_v_deep_q_network2d_16_dense_49_kernel_read_readvariableopCsavev2_adam_m_deep_q_network2d_16_dense_49_bias_read_readvariableopCsavev2_adam_v_deep_q_network2d_16_dense_49_bias_read_readvariableopEsavev2_adam_m_deep_q_network2d_16_dense_50_kernel_read_readvariableopEsavev2_adam_v_deep_q_network2d_16_dense_50_kernel_read_readvariableopCsavev2_adam_m_deep_q_network2d_16_dense_50_bias_read_readvariableopCsavev2_adam_v_deep_q_network2d_16_dense_50_bias_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *#
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*­
_input_shapes
: :@:@:@@:@:@:: : :@:@:@:@:@@:@@:@:@:@:@::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$	 

_output_shapes

:@:$
 

_output_shapes

:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@@:$ 

_output_shapes

:@@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@:$ 

_output_shapes

:@: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 


ї
F__inference_dense_49_layer_call_and_return_conditional_losses_30853171

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ы

+__inference_dense_49_layer_call_fn_30853366

inputs
unknown:@@
	unknown_0:@
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dense_49_layer_call_and_return_conditional_losses_30853171o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs


ї
F__inference_dense_48_layer_call_and_return_conditional_losses_30853357

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Й
P
$__inference__update_step_xla_2493581
gradient
variable:@@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:@@: *
	_noinline(:H D

_output_shapes

:@@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ы

+__inference_dense_50_layer_call_fn_30853386

inputs
unknown:@
	unknown_0:
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_dense_50_layer_call_and_return_conditional_losses_30853187o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
­
L
$__inference__update_step_xla_2493586
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
­
L
$__inference__update_step_xla_2493576
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
	

6__inference_deep_q_network2d_16_layer_call_fn_30853313	
state
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_deep_q_network2d_16_layer_call_and_return_conditional_losses_30853194o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:џџџџџџџџџ

_user_specified_namestate


ї
F__inference_dense_49_layer_call_and_return_conditional_losses_30853377

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
 %
е
#__inference__wrapped_model_30853136
input_1M
;deep_q_network2d_16_dense_48_matmul_readvariableop_resource:@J
<deep_q_network2d_16_dense_48_biasadd_readvariableop_resource:@M
;deep_q_network2d_16_dense_49_matmul_readvariableop_resource:@@J
<deep_q_network2d_16_dense_49_biasadd_readvariableop_resource:@M
;deep_q_network2d_16_dense_50_matmul_readvariableop_resource:@J
<deep_q_network2d_16_dense_50_biasadd_readvariableop_resource:
identityЂ3deep_q_network2d_16/dense_48/BiasAdd/ReadVariableOpЂ2deep_q_network2d_16/dense_48/MatMul/ReadVariableOpЂ3deep_q_network2d_16/dense_49/BiasAdd/ReadVariableOpЂ2deep_q_network2d_16/dense_49/MatMul/ReadVariableOpЂ3deep_q_network2d_16/dense_50/BiasAdd/ReadVariableOpЂ2deep_q_network2d_16/dense_50/MatMul/ReadVariableOpЎ
2deep_q_network2d_16/dense_48/MatMul/ReadVariableOpReadVariableOp;deep_q_network2d_16_dense_48_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Є
#deep_q_network2d_16/dense_48/MatMulMatMulinput_1:deep_q_network2d_16/dense_48/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ќ
3deep_q_network2d_16/dense_48/BiasAdd/ReadVariableOpReadVariableOp<deep_q_network2d_16_dense_48_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Э
$deep_q_network2d_16/dense_48/BiasAddBiasAdd-deep_q_network2d_16/dense_48/MatMul:product:0;deep_q_network2d_16/dense_48/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
!deep_q_network2d_16/dense_48/ReluRelu-deep_q_network2d_16/dense_48/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ў
2deep_q_network2d_16/dense_49/MatMul/ReadVariableOpReadVariableOp;deep_q_network2d_16_dense_49_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0Ь
#deep_q_network2d_16/dense_49/MatMulMatMul/deep_q_network2d_16/dense_48/Relu:activations:0:deep_q_network2d_16/dense_49/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ќ
3deep_q_network2d_16/dense_49/BiasAdd/ReadVariableOpReadVariableOp<deep_q_network2d_16_dense_49_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Э
$deep_q_network2d_16/dense_49/BiasAddBiasAdd-deep_q_network2d_16/dense_49/MatMul:product:0;deep_q_network2d_16/dense_49/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
!deep_q_network2d_16/dense_49/ReluRelu-deep_q_network2d_16/dense_49/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ў
2deep_q_network2d_16/dense_50/MatMul/ReadVariableOpReadVariableOp;deep_q_network2d_16_dense_50_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ь
#deep_q_network2d_16/dense_50/MatMulMatMul/deep_q_network2d_16/dense_49/Relu:activations:0:deep_q_network2d_16/dense_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЌ
3deep_q_network2d_16/dense_50/BiasAdd/ReadVariableOpReadVariableOp<deep_q_network2d_16_dense_50_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Э
$deep_q_network2d_16/dense_50/BiasAddBiasAdd-deep_q_network2d_16/dense_50/MatMul:product:0;deep_q_network2d_16/dense_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ|
IdentityIdentity-deep_q_network2d_16/dense_50/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp4^deep_q_network2d_16/dense_48/BiasAdd/ReadVariableOp3^deep_q_network2d_16/dense_48/MatMul/ReadVariableOp4^deep_q_network2d_16/dense_49/BiasAdd/ReadVariableOp3^deep_q_network2d_16/dense_49/MatMul/ReadVariableOp4^deep_q_network2d_16/dense_50/BiasAdd/ReadVariableOp3^deep_q_network2d_16/dense_50/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 2j
3deep_q_network2d_16/dense_48/BiasAdd/ReadVariableOp3deep_q_network2d_16/dense_48/BiasAdd/ReadVariableOp2h
2deep_q_network2d_16/dense_48/MatMul/ReadVariableOp2deep_q_network2d_16/dense_48/MatMul/ReadVariableOp2j
3deep_q_network2d_16/dense_49/BiasAdd/ReadVariableOp3deep_q_network2d_16/dense_49/BiasAdd/ReadVariableOp2h
2deep_q_network2d_16/dense_49/MatMul/ReadVariableOp2deep_q_network2d_16/dense_49/MatMul/ReadVariableOp2j
3deep_q_network2d_16/dense_50/BiasAdd/ReadVariableOp3deep_q_network2d_16/dense_50/BiasAdd/ReadVariableOp2h
2deep_q_network2d_16/dense_50/MatMul/ReadVariableOp2deep_q_network2d_16/dense_50/MatMul/ReadVariableOp:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1"
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ћ
serving_default
;
input_10
serving_default_input_1:0џџџџџџџџџ<
output_10
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:u

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
fc1
	fc2

fc3
	optimizer
loss

signatures"
_tf_keras_model
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ы
trace_0
trace_12
6__inference_deep_q_network2d_16_layer_call_fn_30853209
6__inference_deep_q_network2d_16_layer_call_fn_30853313Ё
В
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1

trace_0
trace_12Ъ
Q__inference_deep_q_network2d_16_layer_call_and_return_conditional_losses_30853337
Q__inference_deep_q_network2d_16_layer_call_and_return_conditional_losses_30853275Ё
В
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
ЮBЫ
#__inference__wrapped_model_30853136input_1"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Л
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Л
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Л
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer

/
_variables
0_iterations
1_learning_rate
2_index_dict
3
_momentums
4_velocities
5_update_step_xla"
experimentalOptimizer
 "
trackable_dict_wrapper
,
6serving_default"
signature_map
5:3@2#deep_q_network2d_16/dense_48/kernel
/:-@2!deep_q_network2d_16/dense_48/bias
5:3@@2#deep_q_network2d_16/dense_49/kernel
/:-@2!deep_q_network2d_16/dense_49/bias
5:3@2#deep_q_network2d_16/dense_50/kernel
/:-2!deep_q_network2d_16/dense_50/bias
 "
trackable_list_wrapper
5
0
	1

2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ъBч
6__inference_deep_q_network2d_16_layer_call_fn_30853209input_1"Ё
В
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
шBх
6__inference_deep_q_network2d_16_layer_call_fn_30853313state"Ё
В
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
Q__inference_deep_q_network2d_16_layer_call_and_return_conditional_losses_30853337state"Ё
В
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
Q__inference_deep_q_network2d_16_layer_call_and_return_conditional_losses_30853275input_1"Ё
В
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
я
<trace_02в
+__inference_dense_48_layer_call_fn_30853346Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z<trace_0

=trace_02э
F__inference_dense_48_layer_call_and_return_conditional_losses_30853357Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z=trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
я
Ctrace_02в
+__inference_dense_49_layer_call_fn_30853366Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zCtrace_0

Dtrace_02э
F__inference_dense_49_layer_call_and_return_conditional_losses_30853377Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zDtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
я
Jtrace_02в
+__inference_dense_50_layer_call_fn_30853386Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zJtrace_0

Ktrace_02э
F__inference_dense_50_layer_call_and_return_conditional_losses_30853396Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zKtrace_0
~
00
L1
M2
N3
O4
P5
Q6
R7
S8
T9
U10
V11
W12"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
J
L0
N1
P2
R3
T4
V5"
trackable_list_wrapper
J
M0
O1
Q2
S3
U4
W5"
trackable_list_wrapper
П
Xtrace_0
Ytrace_1
Ztrace_2
[trace_3
\trace_4
]trace_52 
$__inference__update_step_xla_2493571
$__inference__update_step_xla_2493576
$__inference__update_step_xla_2493581
$__inference__update_step_xla_2493586
$__inference__update_step_xla_2493591
$__inference__update_step_xla_2493596Й
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0zXtrace_0zYtrace_1zZtrace_2z[trace_3z\trace_4z]trace_5
ЭBЪ
&__inference_signature_wrapper_30853296input_1"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
пBм
+__inference_dense_48_layer_call_fn_30853346inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
F__inference_dense_48_layer_call_and_return_conditional_losses_30853357inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
пBм
+__inference_dense_49_layer_call_fn_30853366inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
F__inference_dense_49_layer_call_and_return_conditional_losses_30853377inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
пBм
+__inference_dense_50_layer_call_fn_30853386inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
F__inference_dense_50_layer_call_and_return_conditional_losses_30853396inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
::8@2*Adam/m/deep_q_network2d_16/dense_48/kernel
::8@2*Adam/v/deep_q_network2d_16/dense_48/kernel
4:2@2(Adam/m/deep_q_network2d_16/dense_48/bias
4:2@2(Adam/v/deep_q_network2d_16/dense_48/bias
::8@@2*Adam/m/deep_q_network2d_16/dense_49/kernel
::8@@2*Adam/v/deep_q_network2d_16/dense_49/kernel
4:2@2(Adam/m/deep_q_network2d_16/dense_49/bias
4:2@2(Adam/v/deep_q_network2d_16/dense_49/bias
::8@2*Adam/m/deep_q_network2d_16/dense_50/kernel
::8@2*Adam/v/deep_q_network2d_16/dense_50/kernel
4:22(Adam/m/deep_q_network2d_16/dense_50/bias
4:22(Adam/v/deep_q_network2d_16/dense_50/bias
љBі
$__inference__update_step_xla_2493571gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
$__inference__update_step_xla_2493576gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
$__inference__update_step_xla_2493581gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
$__inference__update_step_xla_2493586gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
$__inference__update_step_xla_2493591gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
$__inference__update_step_xla_2493596gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
$__inference__update_step_xla_2493571nhЂe
^Ђ[

gradient@
41	Ђ
њ@

p
` VariableSpec 
`рчоРо?
Њ "
 
$__inference__update_step_xla_2493576f`Ђ]
VЂS

gradient@
0-	Ђ
њ@

p
` VariableSpec 
` јоРо?
Њ "
 
$__inference__update_step_xla_2493581nhЂe
^Ђ[

gradient@@
41	Ђ
њ@@

p
` VariableSpec 
`реоРо?
Њ "
 
$__inference__update_step_xla_2493586f`Ђ]
VЂS

gradient@
0-	Ђ
њ@

p
` VariableSpec 
`рРоРо?
Њ "
 
$__inference__update_step_xla_2493591nhЂe
^Ђ[

gradient@
41	Ђ
њ@

p
` VariableSpec 
`рсоРо?
Њ "
 
$__inference__update_step_xla_2493596f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`річщо?
Њ "
 
#__inference__wrapped_model_30853136o0Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ "3Њ0
.
output_1"
output_1џџџџџџџџџН
Q__inference_deep_q_network2d_16_layer_call_and_return_conditional_losses_30853275h0Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 Л
Q__inference_deep_q_network2d_16_layer_call_and_return_conditional_losses_30853337f.Ђ+
$Ђ!

stateџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
6__inference_deep_q_network2d_16_layer_call_fn_30853209]0Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ "!
unknownџџџџџџџџџ
6__inference_deep_q_network2d_16_layer_call_fn_30853313[.Ђ+
$Ђ!

stateџџџџџџџџџ
Њ "!
unknownџџџџџџџџџ­
F__inference_dense_48_layer_call_and_return_conditional_losses_30853357c/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
+__inference_dense_48_layer_call_fn_30853346X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџ@­
F__inference_dense_49_layer_call_and_return_conditional_losses_30853377c/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
+__inference_dense_49_layer_call_fn_30853366X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџ@­
F__inference_dense_50_layer_call_and_return_conditional_losses_30853396c/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
+__inference_dense_50_layer_call_fn_30853386X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџЄ
&__inference_signature_wrapper_30853296z;Ђ8
Ђ 
1Њ.
,
input_1!
input_1џџџџџџџџџ"3Њ0
.
output_1"
output_1џџџџџџџџџ