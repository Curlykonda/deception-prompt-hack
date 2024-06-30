import fastchat
import torch


class SuffixManager_split:
    def __init__(
        self,
        *,
        tokenizer,
        conv_template,
        instruction_1,
        instruction_2,
        target,
        adv_string,
    ):
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.instruction_1 = instruction_1
        self.instruction_2 = instruction_2
        self.target = target
        self.adv_string = adv_string

    def get_prompt(self, adv_string=None):
        if adv_string is not None:
            self.adv_string = adv_string

        self.conv_template.append_message(
            self.conv_template.roles[0],
            f"{self.instruction_1}{self.adv_string}{self.instruction_2}",
        )
        self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
        prompt = self.conv_template.get_prompt()

        encoding = self.tokenizer(prompt)
        toks = encoding.input_ids

        if self.conv_template.name == "llama-2":
            assert False, "This should not happen 2."

            # self.conv_template.messages = []

            # self.conv_template.append_message(self.conv_template.roles[0], None)
            # toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            # self._user_role_slice = slice(None, len(toks))

            # self.conv_template.update_last_message(f"{self.instruction_1}")
            # toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            # self._goal_slice_1 = slice(
            #     self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks))
            # )

            # self.conv_template.update_last_message(
            #     f"{self.instruction_1} {self.adv_string}"
            # )
            # toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            # self._control_slice = slice(self._goal_slice_1.stop, len(toks))

            # self.conv_template.update_last_message(
            #     f"{self.instruction_1} {self.adv_string} {self.instruction_2}"
            # )
            # toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            # self._goal_slice_2 = slice(self._control_slice.stop, len(toks))

            # self.conv_template.append_message(self.conv_template.roles[1], None)
            # toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            # self._assistant_role_slice = slice(self._goal_slice_2.stop, len(toks))

            # self.conv_template.update_last_message(f"{self.target}")
            # toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            # self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 2)
            # self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 3)

        else:
            python_tokenizer = False or self.conv_template.name == "oasst_pythia"
            try:
                encoding.char_to_token(len(prompt) - 1)
            except:
                python_tokenizer = True

            if python_tokenizer:
                self.conv_template.messages = []

                self.conv_template.append_message(self.conv_template.roles[0], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._user_role_slice = slice(None, len(toks))

                self.conv_template.update_last_message(f"{self.instruction_1}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._goal_slice_1 = slice(
                    self._user_role_slice.stop,
                    max(self._user_role_slice.stop, len(toks) - 1),
                )

                self.conv_template.update_last_message(
                    f"{self.instruction_1}{self.adv_string}"
                )
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._control_slice = slice(self._goal_slice_1.stop, len(toks) - 1)

                self.conv_template.update_last_message(
                    f"{self.instruction_1}{self.adv_string}{self.instruction_2}"
                )
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._goal_slice_2 = slice(self._control_slice.stop, len(toks) - 1)

                self.conv_template.append_message(self.conv_template.roles[1], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._assistant_role_slice = slice(self._goal_slice_2.stop, len(toks))

                self.conv_template.update_last_message(f"{self.target}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._target_slice = slice(
                    self._assistant_role_slice.stop, len(toks) - 1
                )
                # self._loss_slice = slice(
                #     self._assistant_role_slice.stop - 1, len(toks) - 2
                # )
                self._loss_slice = slice(self._assistant_role_slice.stop, len(toks) - 1)
            else:
                # self._system_slice = slice(
                #     None, encoding.char_to_token(len(self.conv_template.system))
                # )
                # self._user_role_slice = slice(
                #     encoding.char_to_token(prompt.find(self.conv_template.roles[0])),
                #     encoding.char_to_token(
                #         prompt.find(self.conv_template.roles[0])
                #         + len(self.conv_template.roles[0])
                #         + 1
                #     ),
                # )
                # self._goal_slice_1 = slice(
                #     encoding.char_to_token(prompt.find(self.instruction_1)),
                #     encoding.char_to_token(
                #         prompt.find(self.instruction_1) + len(self.instruction_1)
                #     ),
                # )
                # self._control_slice = slice(
                #     encoding.char_to_token(prompt.find(self.adv_string)),
                #     encoding.char_to_token(
                #         prompt.find(self.adv_string) + len(self.adv_string)
                #     ),
                # )
                # self._goal_slice_2 = slice(
                #     encoding.char_to_token(prompt.find(self.instruction_2)),
                #     encoding.char_to_token(
                #         prompt.find(self.instruction_2) + len(self.instruction_2)
                #     ),
                # )
                # self._assistant_role_slice = slice(
                #     encoding.char_to_token(prompt.find(self.conv_template.roles[1])),
                #     encoding.char_to_token(
                #         prompt.find(self.conv_template.roles[1])
                #         + len(self.conv_template.roles[1])
                #         + 1
                #     ),
                # )
                # self._target_slice = slice(
                #     encoding.char_to_token(prompt.find(self.target)),
                #     encoding.char_to_token(prompt.find(self.target) + len(self.target)),
                # )
                # self._loss_slice = slice(
                #     encoding.char_to_token(prompt.find(self.target)) - 1,
                #     encoding.char_to_token(prompt.find(self.target) + len(self.target))
                #     - 1,
                # )
                assert False, "This should not happen."

        self.conv_template.messages = []

        return prompt

    def get_input_ids(self, adv_string=None):
        prompt = self.get_prompt(adv_string=adv_string)
        toks = self.tokenizer(prompt).input_ids
        input_ids = torch.tensor(toks[: self._target_slice.stop])

        return input_ids


class SuffixManager_split_2:
    def __init__(
        self,
        *,
        tokenizer,
        instruction_1,
        instruction_2,
        target,
        adv_string,
    ):
        self.tokenizer = tokenizer
        self.instruction_1 = instruction_1
        self.instruction_2 = instruction_2
        self.target = target
        self.adv_string = adv_string

    def get_prompt(self, adv_string=None):
        if adv_string is not None:
            self.adv_string = adv_string
        prompt = (
            f"{self.instruction_1}{self.adv_string}{self.instruction_2}{self.target}"
        )
        encoding = self.tokenizer(prompt)
        toks = encoding.input_ids

        self._goal_slice_1 = slice(
            0, len(self.tokenizer(self.instruction_1).input_ids) - 1
        )
        self._control_slice = slice(
            self._goal_slice_1.stop,
            self._goal_slice_1.stop + len(self.tokenizer(self.adv_string).input_ids),
        )
        self._goal_slice_2 = slice(
            self._control_slice.stop,
            self._control_slice.stop
            + len(self.tokenizer(self.instruction_2).input_ids),
        )
        self._target_slice = slice(self._goal_slice_2.stop, len(toks))
        self._loss_slice = self._target_slice

        return prompt

    def get_input_ids(self, adv_string=None):
        prompt = self.get_prompt(adv_string=adv_string)
        toks = self.tokenizer(prompt).input_ids
        input_ids = torch.tensor(toks[: self._target_slice.stop])
        return input_ids
