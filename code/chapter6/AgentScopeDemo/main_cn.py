# -*- coding: utf-8 -*-
"""
三国狼人杀 - 基于AgentScope的中文版狼人杀游戏
融合三国演义角色和传统狼人杀玩法

演示了如何在一个需要实时协作、角色扮演和策略博弈的场景中，充分发挥消息驱动架构的威力

🎭 游戏概述
玩家人数: 通常6-12人(也可更多)
游戏目标:
    好人阵营(村民+神职):找出并投票淘汰所有狼人
    狼人阵营:消灭所有村民或所有神职(屠边规则),或消灭所有好人(屠城规则)

👥 角色分类
🔵 好人阵营
普通村民:没有特殊能力,通过推理和投票找出狼人
预言家:每晚可以查验一名玩家的身份(好人/狼人)
女巫:拥有一瓶解药(救被狼人杀害的玩家)和一瓶毒药(毒死一名玩家),每瓶只能用一次
猎人:被投票出局或被狼人杀害时可以开枪带走一名玩家(被女巫毒杀不能开枪)
守卫:每晚可以守护一名玩家免受狼人杀害(不能连续两晚守护同一人)
白痴:被投票出局时可以翻牌免死,但失去投票权

🔴 狼人阵营
普通狼人:每晚共同选择杀害一名玩家
狼王:被投票出局时可以带走一名玩家
白狼王:白天可以自爆带走一名玩家

🎮 游戏流程
夜晚阶段(所有人闭眼)
狼人行动:狼人睁眼,商量后选择一名玩家杀害
预言家行动:查验一名玩家身份,法官给出手势提示
女巫行动:得知谁被杀害,决定是否使用解药或毒药
守卫行动:选择守护一名玩家
其他神职行动:根据各自技能执行

白天阶段(所有人睁眼)
公布昨夜情况:法官宣布是否有玩家死亡
遗言环节:死亡玩家发表遗言(首夜死亡通常有遗言)
讨论环节:所有存活玩家自由发言,分析推理

投票环节:投票选出一名疑似狼人的玩家
处决:得票最多的玩家出局,发表遗言
进入下一夜:重复上述流程


"""
import asyncio
import os
import random
from typing import List, Dict, Optional

from agentscope.agent import ReActAgent
from agentscope.model import DashScopeChatModel
from agentscope.pipeline import MsgHub, sequential_pipeline, fanout_pipeline
from agentscope.formatter import DashScopeMultiAgentFormatter

from prompt_cn import ChinesePrompts
from game_roles import GameRoles
from structured_output_cn import (
    DiscussionModelCN,
    get_vote_model_cn,
    WitchActionModelCN,
    get_seer_model_cn,
    get_hunter_model_cn,
    WerewolfKillModelCN
)
from utils_cn import (
    check_winning_cn,
    majority_vote_cn,
    get_chinese_name,
    format_player_list,
    GameModerator,
    MAX_GAME_ROUND,
    MAX_DISCUSSION_ROUND,
)


class ThreeKingdomsWerewolfGame:
    """
    # 三国狼人杀游戏主类
    负责维护全局状态（如玩家存活列表、当前游戏阶段）
    推进游戏流程（调用夜晚阶段、白天阶段）
    裁定胜负

    # 智能体交互层
    完全由 MsgHub 驱动。所有智能体间的通信，无论是狼人间的秘密协商，还是白天的公开辩论，都通过消息中心进行路由和分发

    # 角色建模层
    每个玩家都是一个基于 ReactAgent 的实例。我们通过精心设计的系统提示词，为每个智能体注入了“游戏角色”和“三国人格”的双重身份

    """

    def __init__(self):
        self.players: Dict[str, ReActAgent] = {}  # 玩家
        self.roles: Dict[str, str] = {}  # 玩家角色 [name, role]

        self.moderator = GameModerator()  # 游戏主持人

        self.alive_players: List[ReActAgent] = []  # 存活的玩家
        self.werewolves: List[ReActAgent] = []  # 狼人
        self.villagers: List[ReActAgent] = []  # 村民
        self.seer: List[ReActAgent] = []  # 预言家
        self.witch: List[ReActAgent] = []  # 守卫者
        self.hunter: List[ReActAgent] = []  # 猎人

        # 女巫道具状态
        self.witch_has_antidote = True  # 是否有解药
        self.witch_has_poison = True  # 是否有毒药

    async def create_player(self, role: str, character: str) -> ReActAgent:
        """创建具有三国背景的玩家"""
        name = get_chinese_name(character)
        self.roles[name] = role

        # 知识点
        agent = ReActAgent(
            name=name,
            sys_prompt=ChinesePrompts.get_role_prompt(role, character),
            model=DashScopeChatModel(
                model_name="qwen-max",
                api_key=os.environ["DASHSCOPE_API_KEY"],
                enable_thinking=True,
            ),
            formatter=DashScopeMultiAgentFormatter(),
        )

        # 角色身份确认
        # 知识点：agent#observe
        await agent.observe(
            await self.moderator.announce(
                f"【{name}】你在这场三国狼人杀中扮演{GameRoles.get_role_desc(role)}，"
                f"你的角色是{character}。{GameRoles.get_role_ability(role)}"
            )
        )

        self.players[name] = agent
        return agent

    async def setup_game(self, player_count: int = 6):
        """设置游戏"""
        print("🎮 开始设置三国狼人杀游戏...")

        # 获取角色配置
        roles = GameRoles.get_standard_setup(player_count)
        characters = random.sample([
            "刘备", "关羽", "张飞", "诸葛亮", "赵云",
            "曹操", "司马懿", "周瑜", "孙权"
        ], player_count)

        # 创建玩家
        for i, (role, character) in enumerate(zip(roles, characters)):
            agent = await self.create_player(role, character)  # 创建玩家Agent
            self.alive_players.append(agent)

            # 分配到对应阵营
            if role == "狼人":
                self.werewolves.append(agent)
            elif role == "预言家":
                self.seer.append(agent)
            elif role == "女巫":
                self.witch.append(agent)
            elif role == "猎人":
                self.hunter.append(agent)
            else:
                self.villagers.append(agent)

        # 游戏开始公告
        await self.moderator.announce(
            f"三国狼人杀游戏开始！参与者：{format_player_list(self.alive_players)}"
        )

        print(f"✅ 游戏设置完成，共{len(self.alive_players)}名玩家")

    async def werewolf_phase(self, round_num: int):
        """狼人阶段"""
        if not self.werewolves:
            return None

        await self.moderator.announce(f"🐺 狼人请睁眼，选择今晚要击杀的目标...")

        # 狼人讨论
        # 知识点：MsgHub
        # 展示消息驱动的协作模式
        async with MsgHub(
                self.werewolves,
                enable_auto_broadcast=True,
                announcement=await self.moderator.announce(
                    f"狼人们，请讨论今晚的击杀目标。存活玩家：{format_player_list(self.alive_players)}"
                ),
        ) as werewolves_hub:
            # 论阶段
            # 狼人通过消息交换策略
            for _ in range(MAX_DISCUSSION_ROUND):
                for wolf in self.werewolves:
                    await wolf(structured_model=DiscussionModelCN)

            # 投票击杀
            werewolves_hub.set_auto_broadcast(False)
            kill_votes = await fanout_pipeline(
                self.werewolves,
                msg=await self.moderator.announce("请选择击杀目标"),
                structured_model=WerewolfKillModelCN,
                enable_gather=False,
            )

            # 统计投票
            votes = {}
            for i, vote_msg in enumerate(kill_votes):
                # 检查vote_msg是否为None或metadata是否存在
                if vote_msg is not None and hasattr(vote_msg, 'metadata') and vote_msg.metadata is not None:
                    votes[self.werewolves[i].name] = vote_msg.metadata.get("target")
                else:
                    # 如果返回无效,随机选择一个目标
                    print(f"⚠️ {self.werewolves[i].name} 的击杀投票无效,随机选择目标")
                    import random
                    valid_targets = [p.name for p in self.alive_players if
                                     p.name not in [w.name for w in self.werewolves]]
                    votes[self.werewolves[i].name] = random.choice(valid_targets) if valid_targets else None

            killed_player, _ = majority_vote_cn(votes)
            return killed_player

    async def seer_phase(self):
        """预言家阶段"""
        if not self.seer:
            return

        seer_agent = self.seer[0]
        await self.moderator.announce("🔮 预言家请睁眼，选择要查验的玩家...")

        check_result = await seer_agent(
            structured_model=get_seer_model_cn(self.alive_players)
        )

        # 检查返回结果是否有效
        if check_result is None or not hasattr(check_result, 'metadata') or check_result.metadata is None:
            print(f"⚠️ 预言家查验失败,跳过此阶段")
            return

        target_name = check_result.metadata.get("target")
        if not target_name:
            print(f"⚠️ 预言家未选择查验目标,跳过此阶段")
            return

        target_role = self.roles.get(target_name, "村民")

        # 告知预言家结果
        result_msg = f"查验结果：{target_name}是{'狼人' if target_role == '狼人' else '好人'}"
        await seer_agent.observe(await self.moderator.announce(result_msg))

    async def witch_phase(self, killed_player: str):
        """女巫阶段"""
        if not self.witch:
            return killed_player, None

        witch_agent = self.witch[0]
        await self.moderator.announce("🧙‍♀️ 女巫请睁眼...")

        # 告知女巫死亡信息
        death_info = f"今晚{killed_player}被狼人击杀" if killed_player else "今晚平安无事"
        await witch_agent.observe(await self.moderator.announce(death_info))

        # 女巫行动
        witch_action = await witch_agent(structured_model=WitchActionModelCN)

        saved_player = None
        poisoned_player = None

        # 检查返回结果是否有效
        if witch_action is None or not hasattr(witch_action, 'metadata') or witch_action.metadata is None:
            print(f"⚠️ 女巫行动失败,视为不使用技能")
        else:
            if witch_action.metadata.get("use_antidote") and self.witch_has_antidote:
                if killed_player:
                    saved_player = killed_player
                    self.witch_has_antidote = False
                    await witch_agent.observe(await self.moderator.announce(f"你使用解药救了{killed_player}"))

            if witch_action.metadata.get("use_poison") and self.witch_has_poison:
                poisoned_player = witch_action.metadata.get("target_name")
                if poisoned_player:
                    self.witch_has_poison = False
                    await witch_agent.observe(await self.moderator.announce(f"你使用毒药毒杀了{poisoned_player}"))

        # 确定最终死亡玩家
        final_killed = killed_player if not saved_player else None

        return final_killed, poisoned_player

    async def hunter_phase(self, shot_by_hunter: str):
        """猎人阶段"""
        if not self.hunter:
            return None

        hunter_agent = self.hunter[0]
        if hunter_agent.name == shot_by_hunter:
            await self.moderator.announce("🏹 猎人发动技能，可以带走一名玩家...")

            hunter_action = await hunter_agent(
                structured_model=get_hunter_model_cn(self.alive_players)
            )

            # 检查返回结果是否有效
            if hunter_action is None or not hasattr(hunter_action, 'metadata') or hunter_action.metadata is None:
                print(f"⚠️ 猎人技能使用失败,视为放弃开枪")
                return None

            if hunter_action.metadata.get("shoot"):
                target = hunter_action.metadata.get("target")
                if target:
                    await self.moderator.announce(f"猎人{hunter_agent.name}开枪带走了{target}")
                    return target
                else:
                    print(f"⚠️ 猎人选择开枪但未指定目标,视为放弃")
                    return None

        return None

    def update_alive_players(self, dead_players: List[str]):
        """更新存活玩家列表"""
        for dead_name in dead_players:
            if dead_name:
                # 从存活列表移除
                self.alive_players = [p for p in self.alive_players if p.name != dead_name]
                # 从各阵营移除
                self.werewolves = [p for p in self.werewolves if p.name != dead_name]
                self.villagers = [p for p in self.villagers if p.name != dead_name]
                self.seer = [p for p in self.seer if p.name != dead_name]
                self.witch = [p for p in self.witch if p.name != dead_name]
                self.hunter = [p for p in self.hunter if p.name != dead_name]

    async def day_phase(self, round_num: int):
        """白天阶段"""
        await self.moderator.day_announcement(round_num)

        # 讨论阶段
        async with MsgHub(
                self.alive_players,
                enable_auto_broadcast=True,
                announcement=await self.moderator.announce(
                    f"现在开始自由讨论。存活玩家：{format_player_list(self.alive_players)}"
                ),
        ) as all_hub:
            # 每人发言一轮
            # 知识点：sequential_pipeline，顺序执行管道
            await sequential_pipeline(self.alive_players)

            # 投票阶段
            all_hub.set_auto_broadcast(False)
            # 知识点：fanout_pipeline，并行执行管道
            vote_msgs = await fanout_pipeline(
                self.alive_players,
                await self.moderator.announce("请投票选择要淘汰的玩家"),
                structured_model=get_vote_model_cn(self.alive_players),
                enable_gather=False,
            )

            # 统计投票
            votes = {}
            for i, vote_msg in enumerate(vote_msgs):
                # 检查vote_msg是否为None或metadata是否存在
                if vote_msg is not None and hasattr(vote_msg, 'metadata') and vote_msg.metadata is not None:
                    votes[self.alive_players[i].name] = vote_msg.metadata.get("vote")
                else:
                    # 如果返回无效,默认弃票
                    print(f"⚠️ {self.alive_players[i].name} 的投票无效,视为弃票")
                    votes[self.alive_players[i].name] = None

            voted_out, vote_count = majority_vote_cn(votes)
            await self.moderator.vote_result_announcement(voted_out, vote_count)

            return voted_out

    async def run_game(self):
        """运行游戏主循环"""
        try:
            await self.setup_game()

            for round_num in range(1, MAX_GAME_ROUND + 1):
                print(f"\n🌙 === 第{round_num}轮游戏开始 ===")

                # 夜晚阶段
                await self.moderator.night_announcement(round_num)

                # 狼人击杀
                killed_player = await self.werewolf_phase(round_num)

                # 预言家查验
                await self.seer_phase()

                # 女巫行动
                final_killed, poisoned_player = await self.witch_phase(killed_player)

                # 更新死亡玩家
                night_deaths = [p for p in [final_killed, poisoned_player] if p]
                self.update_alive_players(night_deaths)

                # 死亡公告
                await self.moderator.death_announcement(night_deaths)

                # 检查胜利条件
                winner = check_winning_cn(self.alive_players, self.roles)
                if winner:
                    await self.moderator.game_over_announcement(winner)
                    return

                # 白天阶段
                voted_out = await self.day_phase(round_num)

                # 猎人技能
                hunter_shot = await self.hunter_phase(voted_out)

                # 更新死亡玩家
                day_deaths = [p for p in [voted_out, hunter_shot] if p]
                self.update_alive_players(day_deaths)

                # 检查胜利条件
                winner = check_winning_cn(self.alive_players, self.roles)
                if winner:
                    await self.moderator.game_over_announcement(winner)
                    return

                print(f"第{round_num}轮结束，存活玩家：{format_player_list(self.alive_players)}")

        except Exception as e:
            print(f"❌ 游戏运行出错：{e}")
            import traceback
            traceback.print_exc()


async def main():
    """主函数
    🎮 欢迎来到三国狼人杀！
    === 游戏初始化 ===
    游戏主持人: 📢 【孙权】你在这场三国狼人杀中扮演狼人，你的角色是孙权。夜晚可以击杀一名玩家
    游戏主持人: 📢 【周瑜】你在这场三国狼人杀中扮演狼人，你的角色是周瑜。夜晚可以击杀一名玩家
    ...
    游戏主持人: 📢 三国狼人杀游戏开始！参与者：孙权、周瑜、曹操、张飞、司马懿、赵云
    ✅ 游戏设置完成，共6名玩家
    === 第1轮游戏 ===
    🌙 第1夜降临，天黑请闭眼...
    【狼人阶段】
    游戏主持人: 📢 🐺 狼人请睁眼，选择今晚要击杀的目标...
    游戏主持人: 📢 狼人们，请讨论今晚的击杀目标。存活玩家：孙权、周瑜、曹操、张飞、司马懿、赵云
    孙权: 今晚我们应该除掉周瑜，此人智谋过人，对我们威胁很大。
    周瑜: 孙权，你言之有理。但周瑜虽智，却未必是今晚的最大威胁。曹操势力庞大，若不尽早除去，恐对我们不利。
    孙权: 曹操的确是个威胁，但周瑜若活着，他能够识破我们的计谋。不如先解决眼前的隐患。
    周瑜: 孙权，你的顾虑不无道理。但曹操若与我们为敌，他可以联合其他势力对我们构成更大的威胁。
    孙权: 你说的也有道理，曹操的联合确实麻烦。那我们就先对付曹操吧。
    周瑜: 很好，孙权。曹操才是我们今晚首要的目标。
    游戏主持人: 📢 请选择击杀目标
    孙权: 我同意，曹操必须被除掉。
    周瑜: 我同意，曹操是我们今晚要解决的目标。
    【预言家阶段】
    游戏主持人: 📢 🔮 预言家请睁眼，选择要查验的玩家...
    曹操: 我要查验孙权。
    游戏主持人: 📢 查验结果：孙权是狼人
    【女巫阶段】
    游戏主持人: 📢 🧙•♀️ 女巫请睁眼...
    游戏主持人: 📢 今晚曹操被狼人击杀
    张飞: 我昨晚使用了解药救了曹操，现在解药已经用掉了。
    游戏主持人: 📢 你使用解药救了曹操
    游戏主持人: 📢 昨夜平安无事，无人死亡。
    【白天讨论阶段】
    游戏主持人: 📢 ☀️ 第1天天亮了，请大家睁眼...
    游戏主持人: 📢 现在开始自由讨论。存活玩家：孙权、周瑜、曹操、张飞、司马懿、赵云
    孙权: 诸位，曹操势力庞大，对我们都是潜在的威胁。今晚我建议我们集中力量对付他。
    周瑜: 孙权所言极是，曹操不仅自身强大，还可能与其他玩家结盟，对我们构成更大的威胁。
    曹操: 我昨晚查验了孙权，本以为他是好人，但游戏主持人给出的结果却是狼人。这说明有狼人在说谎。
    张飞: 我昨晚确实救了曹操，说明他是被狼人袭击的。但曹操查验孙权的结果令人怀疑。
    司马懿: 曹操的查验结果和张飞的救人行动似乎存在矛盾，我们需要更多的信息来判断谁是狼人。
    赵云: 情况确实复杂，我们需要仔细分析各方的发言。
    【投票阶段】
    游戏主持人: 📢 请投票选择要淘汰的玩家
    孙权: 曹操的威胁依然很大，我坚持认为应该投票给他。
    周瑜: 基于昨晚的情况，我认为我们应该先投票给曹操，他的威胁最大。
    曹操: 我选择投票给孙权。根据游戏主持人的反馈，孙权确实是狼人。
    张飞: 我坚持昨晚救了曹操的事实，但孙权被查出是狼人这一点让我感到困惑。
    司马懿: 我们需要更多的信息来判断谁是狼人。
    [游戏继续...]


    """
    # 检查环境变量
    if "DASHSCOPE_API_KEY" not in os.environ:
        print("❌ 请设置环境变量 DASHSCOPE_API_KEY")
        return

    print("🎮 欢迎来到三国狼人杀！")

    # 创建并运行游戏
    game = ThreeKingdomsWerewolfGame()
    await game.run_game()


if __name__ == "__main__":
    asyncio.run(main())
