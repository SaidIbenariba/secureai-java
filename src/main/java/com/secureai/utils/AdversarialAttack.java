package com.secureai.utils;


import org.nd4j.linalg.api.ndarray.INDArray;

/** Interface for crafting adversarial perturbations on observations. */
    public interface AdversarialAttack {
        INDArray generate(INDArray observation, double epsilon);
    }

