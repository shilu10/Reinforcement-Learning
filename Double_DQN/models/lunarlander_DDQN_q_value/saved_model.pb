ЪЎ
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
 "serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58Ћ
К
1RMSprop/velocity/deep_q_network2d_8/dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31RMSprop/velocity/deep_q_network2d_8/dense_26/bias
Г
ERMSprop/velocity/deep_q_network2d_8/dense_26/bias/Read/ReadVariableOpReadVariableOp1RMSprop/velocity/deep_q_network2d_8/dense_26/bias*
_output_shapes
:*
dtype0
Т
3RMSprop/velocity/deep_q_network2d_8/dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*D
shared_name53RMSprop/velocity/deep_q_network2d_8/dense_26/kernel
Л
GRMSprop/velocity/deep_q_network2d_8/dense_26/kernel/Read/ReadVariableOpReadVariableOp3RMSprop/velocity/deep_q_network2d_8/dense_26/kernel*
_output_shapes

:@*
dtype0
К
1RMSprop/velocity/deep_q_network2d_8/dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*B
shared_name31RMSprop/velocity/deep_q_network2d_8/dense_25/bias
Г
ERMSprop/velocity/deep_q_network2d_8/dense_25/bias/Read/ReadVariableOpReadVariableOp1RMSprop/velocity/deep_q_network2d_8/dense_25/bias*
_output_shapes
:@*
dtype0
Т
3RMSprop/velocity/deep_q_network2d_8/dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*D
shared_name53RMSprop/velocity/deep_q_network2d_8/dense_25/kernel
Л
GRMSprop/velocity/deep_q_network2d_8/dense_25/kernel/Read/ReadVariableOpReadVariableOp3RMSprop/velocity/deep_q_network2d_8/dense_25/kernel*
_output_shapes

:@@*
dtype0
К
1RMSprop/velocity/deep_q_network2d_8/dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*B
shared_name31RMSprop/velocity/deep_q_network2d_8/dense_24/bias
Г
ERMSprop/velocity/deep_q_network2d_8/dense_24/bias/Read/ReadVariableOpReadVariableOp1RMSprop/velocity/deep_q_network2d_8/dense_24/bias*
_output_shapes
:@*
dtype0
Т
3RMSprop/velocity/deep_q_network2d_8/dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*D
shared_name53RMSprop/velocity/deep_q_network2d_8/dense_24/kernel
Л
GRMSprop/velocity/deep_q_network2d_8/dense_24/kernel/Read/ReadVariableOpReadVariableOp3RMSprop/velocity/deep_q_network2d_8/dense_24/kernel*
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

 deep_q_network2d_8/dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" deep_q_network2d_8/dense_26/bias

4deep_q_network2d_8/dense_26/bias/Read/ReadVariableOpReadVariableOp deep_q_network2d_8/dense_26/bias*
_output_shapes
:*
dtype0
 
"deep_q_network2d_8/dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*3
shared_name$"deep_q_network2d_8/dense_26/kernel

6deep_q_network2d_8/dense_26/kernel/Read/ReadVariableOpReadVariableOp"deep_q_network2d_8/dense_26/kernel*
_output_shapes

:@*
dtype0

 deep_q_network2d_8/dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" deep_q_network2d_8/dense_25/bias

4deep_q_network2d_8/dense_25/bias/Read/ReadVariableOpReadVariableOp deep_q_network2d_8/dense_25/bias*
_output_shapes
:@*
dtype0
 
"deep_q_network2d_8/dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*3
shared_name$"deep_q_network2d_8/dense_25/kernel

6deep_q_network2d_8/dense_25/kernel/Read/ReadVariableOpReadVariableOp"deep_q_network2d_8/dense_25/kernel*
_output_shapes

:@@*
dtype0

 deep_q_network2d_8/dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" deep_q_network2d_8/dense_24/bias

4deep_q_network2d_8/dense_24/bias/Read/ReadVariableOpReadVariableOp deep_q_network2d_8/dense_24/bias*
_output_shapes
:@*
dtype0
 
"deep_q_network2d_8/dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*3
shared_name$"deep_q_network2d_8/dense_24/kernel

6deep_q_network2d_8/dense_24/kernel/Read/ReadVariableOpReadVariableOp"deep_q_network2d_8/dense_24/kernel*
_output_shapes

:@*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1"deep_q_network2d_8/dense_24/kernel deep_q_network2d_8/dense_24/bias"deep_q_network2d_8/dense_25/kernel deep_q_network2d_8/dense_25/bias"deep_q_network2d_8/dense_26/kernel deep_q_network2d_8/dense_26/bias*
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
&__inference_signature_wrapper_59060431

NoOpNoOp
ё"
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ќ"
valueЂ"B" B"
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

/
_variables
0_iterations
1_learning_rate
2_index_dict
3_velocities
4
_momentums
5_average_gradients
6_update_step_xla*
* 

7serving_default* 
b\
VARIABLE_VALUE"deep_q_network2d_8/dense_24/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE deep_q_network2d_8/dense_24/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"deep_q_network2d_8/dense_25/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE deep_q_network2d_8/dense_25/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"deep_q_network2d_8/dense_26/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE deep_q_network2d_8/dense_26/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
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
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

=trace_0* 

>trace_0* 

0
1*

0
1*
* 

?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

Dtrace_0* 

Etrace_0* 

0
1*

0
1*
* 

Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

Ktrace_0* 

Ltrace_0* 
5
00
M1
N2
O3
P4
Q5
R6*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
M0
N1
O2
P3
Q4
R5*
* 
* 
P
Strace_0
Ttrace_1
Utrace_2
Vtrace_3
Wtrace_4
Xtrace_5* 
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
~x
VARIABLE_VALUE3RMSprop/velocity/deep_q_network2d_8/dense_24/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE1RMSprop/velocity/deep_q_network2d_8/dense_24/bias1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE3RMSprop/velocity/deep_q_network2d_8/dense_25/kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE1RMSprop/velocity/deep_q_network2d_8/dense_25/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE3RMSprop/velocity/deep_q_network2d_8/dense_26/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE1RMSprop/velocity/deep_q_network2d_8/dense_26/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
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
ь
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename6deep_q_network2d_8/dense_24/kernel/Read/ReadVariableOp4deep_q_network2d_8/dense_24/bias/Read/ReadVariableOp6deep_q_network2d_8/dense_25/kernel/Read/ReadVariableOp4deep_q_network2d_8/dense_25/bias/Read/ReadVariableOp6deep_q_network2d_8/dense_26/kernel/Read/ReadVariableOp4deep_q_network2d_8/dense_26/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpGRMSprop/velocity/deep_q_network2d_8/dense_24/kernel/Read/ReadVariableOpERMSprop/velocity/deep_q_network2d_8/dense_24/bias/Read/ReadVariableOpGRMSprop/velocity/deep_q_network2d_8/dense_25/kernel/Read/ReadVariableOpERMSprop/velocity/deep_q_network2d_8/dense_25/bias/Read/ReadVariableOpGRMSprop/velocity/deep_q_network2d_8/dense_26/kernel/Read/ReadVariableOpERMSprop/velocity/deep_q_network2d_8/dense_26/bias/Read/ReadVariableOpConst*
Tin
2	*
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
!__inference__traced_save_59060596
Я
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename"deep_q_network2d_8/dense_24/kernel deep_q_network2d_8/dense_24/bias"deep_q_network2d_8/dense_25/kernel deep_q_network2d_8/dense_25/bias"deep_q_network2d_8/dense_26/kernel deep_q_network2d_8/dense_26/bias	iterationlearning_rate3RMSprop/velocity/deep_q_network2d_8/dense_24/kernel1RMSprop/velocity/deep_q_network2d_8/dense_24/bias3RMSprop/velocity/deep_q_network2d_8/dense_25/kernel1RMSprop/velocity/deep_q_network2d_8/dense_25/bias3RMSprop/velocity/deep_q_network2d_8/dense_26/kernel1RMSprop/velocity/deep_q_network2d_8/dense_26/bias*
Tin
2*
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
$__inference__traced_restore_59060648Я
	

5__inference_deep_q_network2d_8_layer_call_fn_59060448	
state
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identityЂStatefulPartitionedCall
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
GPU2 *0J 8 *Y
fTRR
P__inference_deep_q_network2d_8_layer_call_and_return_conditional_losses_59060329o
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
Ю

&__inference_signature_wrapper_59060431
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
#__inference__wrapped_model_59060271o
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
Ў
M
%__inference__update_step_xla_31955413
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
	

5__inference_deep_q_network2d_8_layer_call_fn_59060344
input_1
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identityЂStatefulPartitionedCall
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
GPU2 *0J 8 *Y
fTRR
P__inference_deep_q_network2d_8_layer_call_and_return_conditional_losses_59060329o
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
цB
ї

$__inference__traced_restore_59060648
file_prefixE
3assignvariableop_deep_q_network2d_8_dense_24_kernel:@A
3assignvariableop_1_deep_q_network2d_8_dense_24_bias:@G
5assignvariableop_2_deep_q_network2d_8_dense_25_kernel:@@A
3assignvariableop_3_deep_q_network2d_8_dense_25_bias:@G
5assignvariableop_4_deep_q_network2d_8_dense_26_kernel:@A
3assignvariableop_5_deep_q_network2d_8_dense_26_bias:&
assignvariableop_6_iteration:	 *
 assignvariableop_7_learning_rate: X
Fassignvariableop_8_rmsprop_velocity_deep_q_network2d_8_dense_24_kernel:@R
Dassignvariableop_9_rmsprop_velocity_deep_q_network2d_8_dense_24_bias:@Y
Gassignvariableop_10_rmsprop_velocity_deep_q_network2d_8_dense_25_kernel:@@S
Eassignvariableop_11_rmsprop_velocity_deep_q_network2d_8_dense_25_bias:@Y
Gassignvariableop_12_rmsprop_velocity_deep_q_network2d_8_dense_26_kernel:@S
Eassignvariableop_13_rmsprop_velocity_deep_q_network2d_8_dense_26_bias:
identity_15ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*М
valueВBЏB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B щ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOpAssignVariableOp3assignvariableop_deep_q_network2d_8_dense_24_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_1AssignVariableOp3assignvariableop_1_deep_q_network2d_8_dense_24_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_2AssignVariableOp5assignvariableop_2_deep_q_network2d_8_dense_25_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_3AssignVariableOp3assignvariableop_3_deep_q_network2d_8_dense_25_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_4AssignVariableOp5assignvariableop_4_deep_q_network2d_8_dense_26_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_5AssignVariableOp3assignvariableop_5_deep_q_network2d_8_dense_26_biasIdentity_5:output:0"/device:CPU:0*&
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
:н
AssignVariableOp_8AssignVariableOpFassignvariableop_8_rmsprop_velocity_deep_q_network2d_8_dense_24_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_9AssignVariableOpDassignvariableop_9_rmsprop_velocity_deep_q_network2d_8_dense_24_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:р
AssignVariableOp_10AssignVariableOpGassignvariableop_10_rmsprop_velocity_deep_q_network2d_8_dense_25_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_11AssignVariableOpEassignvariableop_11_rmsprop_velocity_deep_q_network2d_8_dense_25_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:р
AssignVariableOp_12AssignVariableOpGassignvariableop_12_rmsprop_velocity_deep_q_network2d_8_dense_26_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_13AssignVariableOpEassignvariableop_13_rmsprop_velocity_deep_q_network2d_8_dense_26_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_15IdentityIdentity_14:output:0^NoOp_1*
T0*
_output_shapes
: №
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_15Identity_15:output:0*1
_input_shapes 
: : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
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
Ы

+__inference_dense_25_layer_call_fn_59060501

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
F__inference_dense_25_layer_call_and_return_conditional_losses_59060306o
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
Щ	
ї
F__inference_dense_26_layer_call_and_return_conditional_losses_59060531

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


ї
F__inference_dense_25_layer_call_and_return_conditional_losses_59060306

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
+__inference_dense_26_layer_call_fn_59060521

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
F__inference_dense_26_layer_call_and_return_conditional_losses_59060322o
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


ї
F__inference_dense_25_layer_call_and_return_conditional_losses_59060512

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
Ў
M
%__inference__update_step_xla_31955403
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
р$
Щ
#__inference__wrapped_model_59060271
input_1L
:deep_q_network2d_8_dense_24_matmul_readvariableop_resource:@I
;deep_q_network2d_8_dense_24_biasadd_readvariableop_resource:@L
:deep_q_network2d_8_dense_25_matmul_readvariableop_resource:@@I
;deep_q_network2d_8_dense_25_biasadd_readvariableop_resource:@L
:deep_q_network2d_8_dense_26_matmul_readvariableop_resource:@I
;deep_q_network2d_8_dense_26_biasadd_readvariableop_resource:
identityЂ2deep_q_network2d_8/dense_24/BiasAdd/ReadVariableOpЂ1deep_q_network2d_8/dense_24/MatMul/ReadVariableOpЂ2deep_q_network2d_8/dense_25/BiasAdd/ReadVariableOpЂ1deep_q_network2d_8/dense_25/MatMul/ReadVariableOpЂ2deep_q_network2d_8/dense_26/BiasAdd/ReadVariableOpЂ1deep_q_network2d_8/dense_26/MatMul/ReadVariableOpЌ
1deep_q_network2d_8/dense_24/MatMul/ReadVariableOpReadVariableOp:deep_q_network2d_8_dense_24_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ђ
"deep_q_network2d_8/dense_24/MatMulMatMulinput_19deep_q_network2d_8/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Њ
2deep_q_network2d_8/dense_24/BiasAdd/ReadVariableOpReadVariableOp;deep_q_network2d_8_dense_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ъ
#deep_q_network2d_8/dense_24/BiasAddBiasAdd,deep_q_network2d_8/dense_24/MatMul:product:0:deep_q_network2d_8/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
 deep_q_network2d_8/dense_24/ReluRelu,deep_q_network2d_8/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ќ
1deep_q_network2d_8/dense_25/MatMul/ReadVariableOpReadVariableOp:deep_q_network2d_8_dense_25_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0Щ
"deep_q_network2d_8/dense_25/MatMulMatMul.deep_q_network2d_8/dense_24/Relu:activations:09deep_q_network2d_8/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Њ
2deep_q_network2d_8/dense_25/BiasAdd/ReadVariableOpReadVariableOp;deep_q_network2d_8_dense_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ъ
#deep_q_network2d_8/dense_25/BiasAddBiasAdd,deep_q_network2d_8/dense_25/MatMul:product:0:deep_q_network2d_8/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
 deep_q_network2d_8/dense_25/ReluRelu,deep_q_network2d_8/dense_25/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ќ
1deep_q_network2d_8/dense_26/MatMul/ReadVariableOpReadVariableOp:deep_q_network2d_8_dense_26_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Щ
"deep_q_network2d_8/dense_26/MatMulMatMul.deep_q_network2d_8/dense_25/Relu:activations:09deep_q_network2d_8/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЊ
2deep_q_network2d_8/dense_26/BiasAdd/ReadVariableOpReadVariableOp;deep_q_network2d_8_dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ъ
#deep_q_network2d_8/dense_26/BiasAddBiasAdd,deep_q_network2d_8/dense_26/MatMul:product:0:deep_q_network2d_8/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ{
IdentityIdentity,deep_q_network2d_8/dense_26/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp3^deep_q_network2d_8/dense_24/BiasAdd/ReadVariableOp2^deep_q_network2d_8/dense_24/MatMul/ReadVariableOp3^deep_q_network2d_8/dense_25/BiasAdd/ReadVariableOp2^deep_q_network2d_8/dense_25/MatMul/ReadVariableOp3^deep_q_network2d_8/dense_26/BiasAdd/ReadVariableOp2^deep_q_network2d_8/dense_26/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 2h
2deep_q_network2d_8/dense_24/BiasAdd/ReadVariableOp2deep_q_network2d_8/dense_24/BiasAdd/ReadVariableOp2f
1deep_q_network2d_8/dense_24/MatMul/ReadVariableOp1deep_q_network2d_8/dense_24/MatMul/ReadVariableOp2h
2deep_q_network2d_8/dense_25/BiasAdd/ReadVariableOp2deep_q_network2d_8/dense_25/BiasAdd/ReadVariableOp2f
1deep_q_network2d_8/dense_25/MatMul/ReadVariableOp1deep_q_network2d_8/dense_25/MatMul/ReadVariableOp2h
2deep_q_network2d_8/dense_26/BiasAdd/ReadVariableOp2deep_q_network2d_8/dense_26/BiasAdd/ReadVariableOp2f
1deep_q_network2d_8/dense_26/MatMul/ReadVariableOp1deep_q_network2d_8/dense_26/MatMul/ReadVariableOp:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
Ф

P__inference_deep_q_network2d_8_layer_call_and_return_conditional_losses_59060472	
state9
'dense_24_matmul_readvariableop_resource:@6
(dense_24_biasadd_readvariableop_resource:@9
'dense_25_matmul_readvariableop_resource:@@6
(dense_25_biasadd_readvariableop_resource:@9
'dense_26_matmul_readvariableop_resource:@6
(dense_26_biasadd_readvariableop_resource:
identityЂdense_24/BiasAdd/ReadVariableOpЂdense_24/MatMul/ReadVariableOpЂdense_25/BiasAdd/ReadVariableOpЂdense_25/MatMul/ReadVariableOpЂdense_26/BiasAdd/ReadVariableOpЂdense_26/MatMul/ReadVariableOp
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0z
dense_24/MatMulMatMulstate&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@b
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0
dense_25/MatMulMatMuldense_24/Relu:activations:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@b
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_26/MatMulMatMuldense_25/Relu:activations:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџh
IdentityIdentitydense_26/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp:N J
'
_output_shapes
:џџџџџџџџџ

_user_specified_namestate
м*
Ф
!__inference__traced_save_59060596
file_prefixA
=savev2_deep_q_network2d_8_dense_24_kernel_read_readvariableop?
;savev2_deep_q_network2d_8_dense_24_bias_read_readvariableopA
=savev2_deep_q_network2d_8_dense_25_kernel_read_readvariableop?
;savev2_deep_q_network2d_8_dense_25_bias_read_readvariableopA
=savev2_deep_q_network2d_8_dense_26_kernel_read_readvariableop?
;savev2_deep_q_network2d_8_dense_26_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableopR
Nsavev2_rmsprop_velocity_deep_q_network2d_8_dense_24_kernel_read_readvariableopP
Lsavev2_rmsprop_velocity_deep_q_network2d_8_dense_24_bias_read_readvariableopR
Nsavev2_rmsprop_velocity_deep_q_network2d_8_dense_25_kernel_read_readvariableopP
Lsavev2_rmsprop_velocity_deep_q_network2d_8_dense_25_bias_read_readvariableopR
Nsavev2_rmsprop_velocity_deep_q_network2d_8_dense_26_kernel_read_readvariableopP
Lsavev2_rmsprop_velocity_deep_q_network2d_8_dense_26_bias_read_readvariableop
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
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*М
valueВBЏB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0=savev2_deep_q_network2d_8_dense_24_kernel_read_readvariableop;savev2_deep_q_network2d_8_dense_24_bias_read_readvariableop=savev2_deep_q_network2d_8_dense_25_kernel_read_readvariableop;savev2_deep_q_network2d_8_dense_25_bias_read_readvariableop=savev2_deep_q_network2d_8_dense_26_kernel_read_readvariableop;savev2_deep_q_network2d_8_dense_26_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableopNsavev2_rmsprop_velocity_deep_q_network2d_8_dense_24_kernel_read_readvariableopLsavev2_rmsprop_velocity_deep_q_network2d_8_dense_24_bias_read_readvariableopNsavev2_rmsprop_velocity_deep_q_network2d_8_dense_25_kernel_read_readvariableopLsavev2_rmsprop_velocity_deep_q_network2d_8_dense_25_bias_read_readvariableopNsavev2_rmsprop_velocity_deep_q_network2d_8_dense_26_kernel_read_readvariableopLsavev2_rmsprop_velocity_deep_q_network2d_8_dense_26_bias_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2	
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

identity_1Identity_1:output:0*{
_input_shapesj
h: :@:@:@@:@:@:: : :@:@:@@:@:@:: 2(
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

:@: 


_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: 


ї
F__inference_dense_24_layer_call_and_return_conditional_losses_59060492

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
К
Q
%__inference__update_step_xla_31955408
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
Ж
Љ
P__inference_deep_q_network2d_8_layer_call_and_return_conditional_losses_59060329	
state#
dense_24_59060290:@
dense_24_59060292:@#
dense_25_59060307:@@
dense_25_59060309:@#
dense_26_59060323:@
dense_26_59060325:
identityЂ dense_24/StatefulPartitionedCallЂ dense_25/StatefulPartitionedCallЂ dense_26/StatefulPartitionedCallњ
 dense_24/StatefulPartitionedCallStatefulPartitionedCallstatedense_24_59060290dense_24_59060292*
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
F__inference_dense_24_layer_call_and_return_conditional_losses_59060289
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_59060307dense_25_59060309*
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
F__inference_dense_25_layer_call_and_return_conditional_losses_59060306
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0dense_26_59060323dense_26_59060325*
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
F__inference_dense_26_layer_call_and_return_conditional_losses_59060322x
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЏ
NoOpNoOp!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall:N J
'
_output_shapes
:џџџџџџџџџ

_user_specified_namestate


ї
F__inference_dense_24_layer_call_and_return_conditional_losses_59060289

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
К
Q
%__inference__update_step_xla_31955388
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
Ў
M
%__inference__update_step_xla_31955393
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
К
Q
%__inference__update_step_xla_31955398
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
Щ	
ї
F__inference_dense_26_layer_call_and_return_conditional_losses_59060322

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
М
Ћ
P__inference_deep_q_network2d_8_layer_call_and_return_conditional_losses_59060410
input_1#
dense_24_59060394:@
dense_24_59060396:@#
dense_25_59060399:@@
dense_25_59060401:@#
dense_26_59060404:@
dense_26_59060406:
identityЂ dense_24/StatefulPartitionedCallЂ dense_25/StatefulPartitionedCallЂ dense_26/StatefulPartitionedCallќ
 dense_24/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_24_59060394dense_24_59060396*
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
F__inference_dense_24_layer_call_and_return_conditional_losses_59060289
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_59060399dense_25_59060401*
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
F__inference_dense_25_layer_call_and_return_conditional_losses_59060306
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0dense_26_59060404dense_26_59060406*
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
F__inference_dense_26_layer_call_and_return_conditional_losses_59060322x
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЏ
NoOpNoOp!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
Ы

+__inference_dense_24_layer_call_fn_59060481

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
F__inference_dense_24_layer_call_and_return_conditional_losses_59060289o
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
 
_user_specified_nameinputs"
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
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:аr
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
Щ
trace_0
trace_12
5__inference_deep_q_network2d_8_layer_call_fn_59060344
5__inference_deep_q_network2d_8_layer_call_fn_59060448Ё
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
џ
trace_0
trace_12Ш
P__inference_deep_q_network2d_8_layer_call_and_return_conditional_losses_59060472
P__inference_deep_q_network2d_8_layer_call_and_return_conditional_losses_59060410Ё
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
#__inference__wrapped_model_59060271input_1"
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
Д
/
_variables
0_iterations
1_learning_rate
2_index_dict
3_velocities
4
_momentums
5_average_gradients
6_update_step_xla"
experimentalOptimizer
 "
trackable_dict_wrapper
,
7serving_default"
signature_map
4:2@2"deep_q_network2d_8/dense_24/kernel
.:,@2 deep_q_network2d_8/dense_24/bias
4:2@@2"deep_q_network2d_8/dense_25/kernel
.:,@2 deep_q_network2d_8/dense_25/bias
4:2@2"deep_q_network2d_8/dense_26/kernel
.:,2 deep_q_network2d_8/dense_26/bias
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
щBц
5__inference_deep_q_network2d_8_layer_call_fn_59060344input_1"Ё
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
чBф
5__inference_deep_q_network2d_8_layer_call_fn_59060448state"Ё
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
Bџ
P__inference_deep_q_network2d_8_layer_call_and_return_conditional_losses_59060472state"Ё
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
B
P__inference_deep_q_network2d_8_layer_call_and_return_conditional_losses_59060410input_1"Ё
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
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
я
=trace_02в
+__inference_dense_24_layer_call_fn_59060481Ђ
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

>trace_02э
F__inference_dense_24_layer_call_and_return_conditional_losses_59060492Ђ
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
 z>trace_0
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
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
я
Dtrace_02в
+__inference_dense_25_layer_call_fn_59060501Ђ
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

Etrace_02э
F__inference_dense_25_layer_call_and_return_conditional_losses_59060512Ђ
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
 zEtrace_0
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
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
я
Ktrace_02в
+__inference_dense_26_layer_call_fn_59060521Ђ
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

Ltrace_02э
F__inference_dense_26_layer_call_and_return_conditional_losses_59060531Ђ
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
 zLtrace_0
Q
00
M1
N2
O3
P4
Q5
R6"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
J
M0
N1
O2
P3
Q4
R5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Х
Strace_0
Ttrace_1
Utrace_2
Vtrace_3
Wtrace_4
Xtrace_52І
%__inference__update_step_xla_31955388
%__inference__update_step_xla_31955393
%__inference__update_step_xla_31955398
%__inference__update_step_xla_31955403
%__inference__update_step_xla_31955408
%__inference__update_step_xla_31955413Й
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
 0zStrace_0zTtrace_1zUtrace_2zVtrace_3zWtrace_4zXtrace_5
ЭBЪ
&__inference_signature_wrapper_59060431input_1"
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
+__inference_dense_24_layer_call_fn_59060481inputs"Ђ
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
F__inference_dense_24_layer_call_and_return_conditional_losses_59060492inputs"Ђ
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
+__inference_dense_25_layer_call_fn_59060501inputs"Ђ
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
F__inference_dense_25_layer_call_and_return_conditional_losses_59060512inputs"Ђ
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
+__inference_dense_26_layer_call_fn_59060521inputs"Ђ
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
F__inference_dense_26_layer_call_and_return_conditional_losses_59060531inputs"Ђ
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
C:A@23RMSprop/velocity/deep_q_network2d_8/dense_24/kernel
=:;@21RMSprop/velocity/deep_q_network2d_8/dense_24/bias
C:A@@23RMSprop/velocity/deep_q_network2d_8/dense_25/kernel
=:;@21RMSprop/velocity/deep_q_network2d_8/dense_25/bias
C:A@23RMSprop/velocity/deep_q_network2d_8/dense_26/kernel
=:;21RMSprop/velocity/deep_q_network2d_8/dense_26/bias
њBї
%__inference__update_step_xla_31955388gradientvariable"З
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
њBї
%__inference__update_step_xla_31955393gradientvariable"З
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
њBї
%__inference__update_step_xla_31955398gradientvariable"З
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
њBї
%__inference__update_step_xla_31955403gradientvariable"З
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
њBї
%__inference__update_step_xla_31955408gradientvariable"З
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
њBї
%__inference__update_step_xla_31955413gradientvariable"З
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
 
%__inference__update_step_xla_31955388nhЂe
^Ђ[

gradient@
41	Ђ
њ@

p
` VariableSpec 
` уЛШяе?
Њ "
 
%__inference__update_step_xla_31955393f`Ђ]
VЂS

gradient@
0-	Ђ
њ@

p
` VariableSpec 
`р№ЛШяе?
Њ "
 
%__inference__update_step_xla_31955398nhЂe
^Ђ[

gradient@@
41	Ђ
њ@@

p
` VariableSpec 
` јЛШяе?
Њ "
 
%__inference__update_step_xla_31955403f`Ђ]
VЂS

gradient@
0-	Ђ
њ@

p
` VariableSpec 
` ўЛШяе?
Њ "
 
%__inference__update_step_xla_31955408nhЂe
^Ђ[

gradient@
41	Ђ
њ@

p
` VariableSpec 
`рЦЁШяе?
Њ "
 
%__inference__update_step_xla_31955413f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рЬЁШяе?
Њ "
 
#__inference__wrapped_model_59060271o0Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ "3Њ0
.
output_1"
output_1џџџџџџџџџМ
P__inference_deep_q_network2d_8_layer_call_and_return_conditional_losses_59060410h0Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 К
P__inference_deep_q_network2d_8_layer_call_and_return_conditional_losses_59060472f.Ђ+
$Ђ!

stateџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
5__inference_deep_q_network2d_8_layer_call_fn_59060344]0Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ "!
unknownџџџџџџџџџ
5__inference_deep_q_network2d_8_layer_call_fn_59060448[.Ђ+
$Ђ!

stateџџџџџџџџџ
Њ "!
unknownџџџџџџџџџ­
F__inference_dense_24_layer_call_and_return_conditional_losses_59060492c/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
+__inference_dense_24_layer_call_fn_59060481X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџ@­
F__inference_dense_25_layer_call_and_return_conditional_losses_59060512c/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
+__inference_dense_25_layer_call_fn_59060501X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџ@­
F__inference_dense_26_layer_call_and_return_conditional_losses_59060531c/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
+__inference_dense_26_layer_call_fn_59060521X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџЄ
&__inference_signature_wrapper_59060431z;Ђ8
Ђ 
1Њ.
,
input_1!
input_1џџџџџџџџџ"3Њ0
.
output_1"
output_1џџџџџџџџџ