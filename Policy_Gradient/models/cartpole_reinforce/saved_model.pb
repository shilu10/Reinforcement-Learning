��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
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
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
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
executor_typestring ��
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58��
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
�
"policy_network2d_47/dense_143/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"policy_network2d_47/dense_143/bias
�
6policy_network2d_47/dense_143/bias/Read/ReadVariableOpReadVariableOp"policy_network2d_47/dense_143/bias*
_output_shapes
:*
dtype0
�
$policy_network2d_47/dense_143/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$*5
shared_name&$policy_network2d_47/dense_143/kernel
�
8policy_network2d_47/dense_143/kernel/Read/ReadVariableOpReadVariableOp$policy_network2d_47/dense_143/kernel*
_output_shapes

:$*
dtype0
�
"policy_network2d_47/dense_142/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*3
shared_name$"policy_network2d_47/dense_142/bias
�
6policy_network2d_47/dense_142/bias/Read/ReadVariableOpReadVariableOp"policy_network2d_47/dense_142/bias*
_output_shapes
:$*
dtype0
�
$policy_network2d_47/dense_142/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$*5
shared_name&$policy_network2d_47/dense_142/kernel
�
8policy_network2d_47/dense_142/kernel/Read/ReadVariableOpReadVariableOp$policy_network2d_47/dense_142/kernel*
_output_shapes

:$*
dtype0
�
"policy_network2d_47/dense_141/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"policy_network2d_47/dense_141/bias
�
6policy_network2d_47/dense_141/bias/Read/ReadVariableOpReadVariableOp"policy_network2d_47/dense_141/bias*
_output_shapes
:*
dtype0
�
$policy_network2d_47/dense_141/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$policy_network2d_47/dense_141/kernel
�
8policy_network2d_47/dense_141/kernel/Read/ReadVariableOpReadVariableOp$policy_network2d_47/dense_141/kernel*
_output_shapes

:*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1$policy_network2d_47/dense_141/kernel"policy_network2d_47/dense_141/bias$policy_network2d_47/dense_142/kernel"policy_network2d_47/dense_142/bias$policy_network2d_47/dense_143/kernel"policy_network2d_47/dense_143/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� */
f*R(
&__inference_signature_wrapper_23516565

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
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
�
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
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

kernel
bias*
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

kernel
bias*
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

kernel
bias*
O
/
_variables
0_iterations
1_learning_rate
2_update_step_xla*
* 

3serving_default* 
d^
VARIABLE_VALUE$policy_network2d_47/dense_141/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"policy_network2d_47/dense_141/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$policy_network2d_47/dense_142/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"policy_network2d_47/dense_142/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$policy_network2d_47/dense_143/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"policy_network2d_47/dense_143/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
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
�
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

9trace_0* 

:trace_0* 

0
1*

0
1*
* 
�
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

@trace_0* 

Atrace_0* 

0
1*

0
1*
* 
�
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

Gtrace_0* 

Htrace_0* 

00*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
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
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename8policy_network2d_47/dense_141/kernel/Read/ReadVariableOp6policy_network2d_47/dense_141/bias/Read/ReadVariableOp8policy_network2d_47/dense_142/kernel/Read/ReadVariableOp6policy_network2d_47/dense_142/bias/Read/ReadVariableOp8policy_network2d_47/dense_143/kernel/Read/ReadVariableOp6policy_network2d_47/dense_143/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpConst*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_save_23516714
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename$policy_network2d_47/dense_141/kernel"policy_network2d_47/dense_141/bias$policy_network2d_47/dense_142/kernel"policy_network2d_47/dense_142/bias$policy_network2d_47/dense_143/kernel"policy_network2d_47/dense_143/bias	iterationlearning_rate*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference__traced_restore_23516748��
�

�
G__inference_dense_142_layer_call_and_return_conditional_losses_23516647

inputs0
matmul_readvariableop_resource:$-
biasadd_readvariableop_resource:$
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������$a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������$w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
G__inference_dense_141_layer_call_and_return_conditional_losses_23516422

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
G__inference_dense_143_layer_call_and_return_conditional_losses_23516667

inputs0
matmul_readvariableop_resource:$-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
�
Q__inference_policy_network2d_47_layer_call_and_return_conditional_losses_23516607	
state:
(dense_141_matmul_readvariableop_resource:7
)dense_141_biasadd_readvariableop_resource::
(dense_142_matmul_readvariableop_resource:$7
)dense_142_biasadd_readvariableop_resource:$:
(dense_143_matmul_readvariableop_resource:$7
)dense_143_biasadd_readvariableop_resource:
identity�� dense_141/BiasAdd/ReadVariableOp�dense_141/MatMul/ReadVariableOp� dense_142/BiasAdd/ReadVariableOp�dense_142/MatMul/ReadVariableOp� dense_143/BiasAdd/ReadVariableOp�dense_143/MatMul/ReadVariableOp�
dense_141/MatMul/ReadVariableOpReadVariableOp(dense_141_matmul_readvariableop_resource*
_output_shapes

:*
dtype0|
dense_141/MatMulMatMulstate'dense_141/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_141/BiasAdd/ReadVariableOpReadVariableOp)dense_141_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_141/BiasAddBiasAdddense_141/MatMul:product:0(dense_141/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d
dense_141/ReluReludense_141/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_142/MatMul/ReadVariableOpReadVariableOp(dense_142_matmul_readvariableop_resource*
_output_shapes

:$*
dtype0�
dense_142/MatMulMatMuldense_141/Relu:activations:0'dense_142/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$�
 dense_142/BiasAdd/ReadVariableOpReadVariableOp)dense_142_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0�
dense_142/BiasAddBiasAdddense_142/MatMul:product:0(dense_142/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$d
dense_142/ReluReludense_142/BiasAdd:output:0*
T0*'
_output_shapes
:���������$�
dense_143/MatMul/ReadVariableOpReadVariableOp(dense_143_matmul_readvariableop_resource*
_output_shapes

:$*
dtype0�
dense_143/MatMulMatMuldense_142/Relu:activations:0'dense_143/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_143/BiasAdd/ReadVariableOpReadVariableOp)dense_143_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_143/BiasAddBiasAdddense_143/MatMul:product:0(dense_143/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������j
dense_143/SoftmaxSoftmaxdense_143/BiasAdd:output:0*
T0*'
_output_shapes
:���������j
IdentityIdentitydense_143/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_141/BiasAdd/ReadVariableOp ^dense_141/MatMul/ReadVariableOp!^dense_142/BiasAdd/ReadVariableOp ^dense_142/MatMul/ReadVariableOp!^dense_143/BiasAdd/ReadVariableOp ^dense_143/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2D
 dense_141/BiasAdd/ReadVariableOp dense_141/BiasAdd/ReadVariableOp2B
dense_141/MatMul/ReadVariableOpdense_141/MatMul/ReadVariableOp2D
 dense_142/BiasAdd/ReadVariableOp dense_142/BiasAdd/ReadVariableOp2B
dense_142/MatMul/ReadVariableOpdense_142/MatMul/ReadVariableOp2D
 dense_143/BiasAdd/ReadVariableOp dense_143/BiasAdd/ReadVariableOp2B
dense_143/MatMul/ReadVariableOpdense_143/MatMul/ReadVariableOp:N J
'
_output_shapes
:���������

_user_specified_namestate
�

�
G__inference_dense_143_layer_call_and_return_conditional_losses_23516456

inputs0
matmul_readvariableop_resource:$-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�	
�
6__inference_policy_network2d_47_layer_call_fn_23516582	
state
unknown:
	unknown_0:
	unknown_1:$
	unknown_2:$
	unknown_3:$
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_policy_network2d_47_layer_call_and_return_conditional_losses_23516463o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:���������

_user_specified_namestate
�
�
,__inference_dense_142_layer_call_fn_23516636

inputs
unknown:$
	unknown_0:$
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_142_layer_call_and_return_conditional_losses_23516439o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
G__inference_dense_141_layer_call_and_return_conditional_losses_23516627

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
G__inference_dense_142_layer_call_and_return_conditional_losses_23516439

inputs0
matmul_readvariableop_resource:$-
biasadd_readvariableop_resource:$
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������$a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������$w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
Q__inference_policy_network2d_47_layer_call_and_return_conditional_losses_23516463	
state$
dense_141_23516423: 
dense_141_23516425:$
dense_142_23516440:$ 
dense_142_23516442:$$
dense_143_23516457:$ 
dense_143_23516459:
identity��!dense_141/StatefulPartitionedCall�!dense_142/StatefulPartitionedCall�!dense_143/StatefulPartitionedCall�
!dense_141/StatefulPartitionedCallStatefulPartitionedCallstatedense_141_23516423dense_141_23516425*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_141_layer_call_and_return_conditional_losses_23516422�
!dense_142/StatefulPartitionedCallStatefulPartitionedCall*dense_141/StatefulPartitionedCall:output:0dense_142_23516440dense_142_23516442*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_142_layer_call_and_return_conditional_losses_23516439�
!dense_143/StatefulPartitionedCallStatefulPartitionedCall*dense_142/StatefulPartitionedCall:output:0dense_143_23516457dense_143_23516459*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_143_layer_call_and_return_conditional_losses_23516456y
IdentityIdentity*dense_143/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_141/StatefulPartitionedCall"^dense_142/StatefulPartitionedCall"^dense_143/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2F
!dense_141/StatefulPartitionedCall!dense_141/StatefulPartitionedCall2F
!dense_142/StatefulPartitionedCall!dense_142/StatefulPartitionedCall2F
!dense_143/StatefulPartitionedCall!dense_143/StatefulPartitionedCall:N J
'
_output_shapes
:���������

_user_specified_namestate
�
�
,__inference_dense_143_layer_call_fn_23516656

inputs
unknown:$
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_143_layer_call_and_return_conditional_losses_23516456o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������$: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
�
&__inference_signature_wrapper_23516565
input_1
unknown:
	unknown_0:
	unknown_1:$
	unknown_2:$
	unknown_3:$
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference__wrapped_model_23516404o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�	
�
6__inference_policy_network2d_47_layer_call_fn_23516478
input_1
unknown:
	unknown_0:
	unknown_1:$
	unknown_2:$
	unknown_3:$
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_policy_network2d_47_layer_call_and_return_conditional_losses_23516463o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�&
�
#__inference__wrapped_model_23516404
input_1N
<policy_network2d_47_dense_141_matmul_readvariableop_resource:K
=policy_network2d_47_dense_141_biasadd_readvariableop_resource:N
<policy_network2d_47_dense_142_matmul_readvariableop_resource:$K
=policy_network2d_47_dense_142_biasadd_readvariableop_resource:$N
<policy_network2d_47_dense_143_matmul_readvariableop_resource:$K
=policy_network2d_47_dense_143_biasadd_readvariableop_resource:
identity��4policy_network2d_47/dense_141/BiasAdd/ReadVariableOp�3policy_network2d_47/dense_141/MatMul/ReadVariableOp�4policy_network2d_47/dense_142/BiasAdd/ReadVariableOp�3policy_network2d_47/dense_142/MatMul/ReadVariableOp�4policy_network2d_47/dense_143/BiasAdd/ReadVariableOp�3policy_network2d_47/dense_143/MatMul/ReadVariableOp�
3policy_network2d_47/dense_141/MatMul/ReadVariableOpReadVariableOp<policy_network2d_47_dense_141_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
$policy_network2d_47/dense_141/MatMulMatMulinput_1;policy_network2d_47/dense_141/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
4policy_network2d_47/dense_141/BiasAdd/ReadVariableOpReadVariableOp=policy_network2d_47_dense_141_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
%policy_network2d_47/dense_141/BiasAddBiasAdd.policy_network2d_47/dense_141/MatMul:product:0<policy_network2d_47/dense_141/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"policy_network2d_47/dense_141/ReluRelu.policy_network2d_47/dense_141/BiasAdd:output:0*
T0*'
_output_shapes
:����������
3policy_network2d_47/dense_142/MatMul/ReadVariableOpReadVariableOp<policy_network2d_47_dense_142_matmul_readvariableop_resource*
_output_shapes

:$*
dtype0�
$policy_network2d_47/dense_142/MatMulMatMul0policy_network2d_47/dense_141/Relu:activations:0;policy_network2d_47/dense_142/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$�
4policy_network2d_47/dense_142/BiasAdd/ReadVariableOpReadVariableOp=policy_network2d_47_dense_142_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0�
%policy_network2d_47/dense_142/BiasAddBiasAdd.policy_network2d_47/dense_142/MatMul:product:0<policy_network2d_47/dense_142/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$�
"policy_network2d_47/dense_142/ReluRelu.policy_network2d_47/dense_142/BiasAdd:output:0*
T0*'
_output_shapes
:���������$�
3policy_network2d_47/dense_143/MatMul/ReadVariableOpReadVariableOp<policy_network2d_47_dense_143_matmul_readvariableop_resource*
_output_shapes

:$*
dtype0�
$policy_network2d_47/dense_143/MatMulMatMul0policy_network2d_47/dense_142/Relu:activations:0;policy_network2d_47/dense_143/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
4policy_network2d_47/dense_143/BiasAdd/ReadVariableOpReadVariableOp=policy_network2d_47_dense_143_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
%policy_network2d_47/dense_143/BiasAddBiasAdd.policy_network2d_47/dense_143/MatMul:product:0<policy_network2d_47/dense_143/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
%policy_network2d_47/dense_143/SoftmaxSoftmax.policy_network2d_47/dense_143/BiasAdd:output:0*
T0*'
_output_shapes
:���������~
IdentityIdentity/policy_network2d_47/dense_143/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp5^policy_network2d_47/dense_141/BiasAdd/ReadVariableOp4^policy_network2d_47/dense_141/MatMul/ReadVariableOp5^policy_network2d_47/dense_142/BiasAdd/ReadVariableOp4^policy_network2d_47/dense_142/MatMul/ReadVariableOp5^policy_network2d_47/dense_143/BiasAdd/ReadVariableOp4^policy_network2d_47/dense_143/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2l
4policy_network2d_47/dense_141/BiasAdd/ReadVariableOp4policy_network2d_47/dense_141/BiasAdd/ReadVariableOp2j
3policy_network2d_47/dense_141/MatMul/ReadVariableOp3policy_network2d_47/dense_141/MatMul/ReadVariableOp2l
4policy_network2d_47/dense_142/BiasAdd/ReadVariableOp4policy_network2d_47/dense_142/BiasAdd/ReadVariableOp2j
3policy_network2d_47/dense_142/MatMul/ReadVariableOp3policy_network2d_47/dense_142/MatMul/ReadVariableOp2l
4policy_network2d_47/dense_143/BiasAdd/ReadVariableOp4policy_network2d_47/dense_143/BiasAdd/ReadVariableOp2j
3policy_network2d_47/dense_143/MatMul/ReadVariableOp3policy_network2d_47/dense_143/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
,__inference_dense_141_layer_call_fn_23516616

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_141_layer_call_and_return_conditional_losses_23516422o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�'
�
$__inference__traced_restore_23516748
file_prefixG
5assignvariableop_policy_network2d_47_dense_141_kernel:C
5assignvariableop_1_policy_network2d_47_dense_141_bias:I
7assignvariableop_2_policy_network2d_47_dense_142_kernel:$C
5assignvariableop_3_policy_network2d_47_dense_142_bias:$I
7assignvariableop_4_policy_network2d_47_dense_143_kernel:$C
5assignvariableop_5_policy_network2d_47_dense_143_bias:&
assignvariableop_6_iteration:	 *
 assignvariableop_7_learning_rate: 

identity_9��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp5assignvariableop_policy_network2d_47_dense_141_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp5assignvariableop_1_policy_network2d_47_dense_141_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp7assignvariableop_2_policy_network2d_47_dense_142_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp5assignvariableop_3_policy_network2d_47_dense_142_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp7assignvariableop_4_policy_network2d_47_dense_143_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp5assignvariableop_5_policy_network2d_47_dense_143_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_iterationIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_9IdentityIdentity_8:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*"
_acd_function_control_output(*
_output_shapes
 "!

identity_9Identity_9:output:0*%
_input_shapes
: : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
Q__inference_policy_network2d_47_layer_call_and_return_conditional_losses_23516544
input_1$
dense_141_23516528: 
dense_141_23516530:$
dense_142_23516533:$ 
dense_142_23516535:$$
dense_143_23516538:$ 
dense_143_23516540:
identity��!dense_141/StatefulPartitionedCall�!dense_142/StatefulPartitionedCall�!dense_143/StatefulPartitionedCall�
!dense_141/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_141_23516528dense_141_23516530*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_141_layer_call_and_return_conditional_losses_23516422�
!dense_142/StatefulPartitionedCallStatefulPartitionedCall*dense_141/StatefulPartitionedCall:output:0dense_142_23516533dense_142_23516535*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_142_layer_call_and_return_conditional_losses_23516439�
!dense_143/StatefulPartitionedCallStatefulPartitionedCall*dense_142/StatefulPartitionedCall:output:0dense_143_23516538dense_143_23516540*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_143_layer_call_and_return_conditional_losses_23516456y
IdentityIdentity*dense_143/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_141/StatefulPartitionedCall"^dense_142/StatefulPartitionedCall"^dense_143/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2F
!dense_141/StatefulPartitionedCall!dense_141/StatefulPartitionedCall2F
!dense_142/StatefulPartitionedCall!dense_142/StatefulPartitionedCall2F
!dense_143/StatefulPartitionedCall!dense_143/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
!__inference__traced_save_23516714
file_prefixC
?savev2_policy_network2d_47_dense_141_kernel_read_readvariableopA
=savev2_policy_network2d_47_dense_141_bias_read_readvariableopC
?savev2_policy_network2d_47_dense_142_kernel_read_readvariableopA
=savev2_policy_network2d_47_dense_142_bias_read_readvariableopC
?savev2_policy_network2d_47_dense_143_kernel_read_readvariableopA
=savev2_policy_network2d_47_dense_143_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0?savev2_policy_network2d_47_dense_141_kernel_read_readvariableop=savev2_policy_network2d_47_dense_141_bias_read_readvariableop?savev2_policy_network2d_47_dense_142_kernel_read_readvariableop=savev2_policy_network2d_47_dense_142_bias_read_readvariableop?savev2_policy_network2d_47_dense_143_kernel_read_readvariableop=savev2_policy_network2d_47_dense_143_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2		�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
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

identity_1Identity_1:output:0*K
_input_shapes:
8: :::$:$:$:: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:$: 

_output_shapes
:$:$ 

_output_shapes

:$: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: "�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:�W
�
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
�
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
�
trace_0
trace_12�
6__inference_policy_network2d_47_layer_call_fn_23516478
6__inference_policy_network2d_47_layer_call_fn_23516582�
���
FullArgSpec
args�
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1
�
trace_0
trace_12�
Q__inference_policy_network2d_47_layer_call_and_return_conditional_losses_23516607
Q__inference_policy_network2d_47_layer_call_and_return_conditional_losses_23516544�
���
FullArgSpec
args�
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1
�B�
#__inference__wrapped_model_23516404input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
j
/
_variables
0_iterations
1_learning_rate
2_update_step_xla"
experimentalOptimizer
 "
trackable_dict_wrapper
,
3serving_default"
signature_map
6:42$policy_network2d_47/dense_141/kernel
0:.2"policy_network2d_47/dense_141/bias
6:4$2$policy_network2d_47/dense_142/kernel
0:.$2"policy_network2d_47/dense_142/bias
6:4$2$policy_network2d_47/dense_143/kernel
0:.2"policy_network2d_47/dense_143/bias
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
�B�
6__inference_policy_network2d_47_layer_call_fn_23516478input_1"�
���
FullArgSpec
args�
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_policy_network2d_47_layer_call_fn_23516582state"�
���
FullArgSpec
args�
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_policy_network2d_47_layer_call_and_return_conditional_losses_23516607state"�
���
FullArgSpec
args�
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_policy_network2d_47_layer_call_and_return_conditional_losses_23516544input_1"�
���
FullArgSpec
args�
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�
9trace_02�
,__inference_dense_141_layer_call_fn_23516616�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z9trace_0
�
:trace_02�
G__inference_dense_141_layer_call_and_return_conditional_losses_23516627�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z:trace_0
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
�
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
�
@trace_02�
,__inference_dense_142_layer_call_fn_23516636�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z@trace_0
�
Atrace_02�
G__inference_dense_142_layer_call_and_return_conditional_losses_23516647�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zAtrace_0
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
�
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�
Gtrace_02�
,__inference_dense_143_layer_call_fn_23516656�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zGtrace_0
�
Htrace_02�
G__inference_dense_143_layer_call_and_return_conditional_losses_23516667�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zHtrace_0
'
00"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
�2��
���
FullArgSpec2
args*�'
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
&__inference_signature_wrapper_23516565input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
,__inference_dense_141_layer_call_fn_23516616inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dense_141_layer_call_and_return_conditional_losses_23516627inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
,__inference_dense_142_layer_call_fn_23516636inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dense_142_layer_call_and_return_conditional_losses_23516647inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
,__inference_dense_143_layer_call_fn_23516656inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dense_143_layer_call_and_return_conditional_losses_23516667inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
#__inference__wrapped_model_23516404o0�-
&�#
!�
input_1���������
� "3�0
.
output_1"�
output_1����������
G__inference_dense_141_layer_call_and_return_conditional_losses_23516627c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
,__inference_dense_141_layer_call_fn_23516616X/�,
%�"
 �
inputs���������
� "!�
unknown����������
G__inference_dense_142_layer_call_and_return_conditional_losses_23516647c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������$
� �
,__inference_dense_142_layer_call_fn_23516636X/�,
%�"
 �
inputs���������
� "!�
unknown���������$�
G__inference_dense_143_layer_call_and_return_conditional_losses_23516667c/�,
%�"
 �
inputs���������$
� ",�)
"�
tensor_0���������
� �
,__inference_dense_143_layer_call_fn_23516656X/�,
%�"
 �
inputs���������$
� "!�
unknown����������
Q__inference_policy_network2d_47_layer_call_and_return_conditional_losses_23516544h0�-
&�#
!�
input_1���������
� ",�)
"�
tensor_0���������
� �
Q__inference_policy_network2d_47_layer_call_and_return_conditional_losses_23516607f.�+
$�!
�
state���������
� ",�)
"�
tensor_0���������
� �
6__inference_policy_network2d_47_layer_call_fn_23516478]0�-
&�#
!�
input_1���������
� "!�
unknown����������
6__inference_policy_network2d_47_layer_call_fn_23516582[.�+
$�!
�
state���������
� "!�
unknown����������
&__inference_signature_wrapper_23516565z;�8
� 
1�.
,
input_1!�
input_1���������"3�0
.
output_1"�
output_1���������