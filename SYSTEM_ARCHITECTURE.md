### No-Le-DRL

## Here a the system architect description for No-Le-DRL. 

# The database tables and fields will be defined in `drl_database.db`, note that the date and time will follow ISO format and use UTC+7 time region (Asia, Ho Chi Minh City). Lists of values are stored as JSON-encoded `TEXT` arrays in SQLite.

1. `events` table: contains list of events collected by the server.
   - `event_id` (INT, primary key): id for event record
   - `title` (TEXT): the extracted title
   - `description` (TEXT): the summerized description of the event's content and benefits for participants.
   - `keywords` (TEXT, JSON list of TEXT): list of keywords extracted directly from the raw text that describe the event instead of a whole sentence (used for the classification module).
   - `is_online` (BOOL): mark whether the event is held online.
   - `reg_start_datetime` (TEXT): the date and time that the event opens for registrations.
   - `reg_end_datetime` (TEXT): the date and time that the event stops allowing registrations.
   - `event_start_datetime` (TEXT): the date and time that the event occurs.
   - `event_end_datetime` (TEXT): the date and time that the event ends.
   - `location` (TEXT): the location that the event takes place.
   - `coordinate` (TEXT): the map coordinate of the simplified location.
   - `fee` (TEXT): the fee of the event.
   - `target_audience` (TEXT): the target audience for the event.
   - `topic` (TEXT): the topics to decide which category the event will be classified as (academic, sports, community, hirtorical, coding, geography,...).
   - `format` (TEXT): the format of how the event is held and how would participants take part in (workshop, seminar, volunteering, contest, field trip, survey,...).
   - `award` (TEXT): how participants would be awarded if attend or complete a specific task (bonus point, certificates,...).
   - `created_at` (TEXT): the date and time where the record is added.
   - `updated_at` (TEXT): the date and time where the record is last updated.
   - `sources` (TEXT): facebook fanpages name or email from organizer or both.
   - `post_url` (TEXT): the facebook post link if it was from a facebook post.
   - `images_url` (TEXT, JSON list of TEXT): media image urls if it was from a facebook post.
   - `category` (INT): the category in the school's rubric that the event would fit the most.
   - `min_credit` (INT): the minimum estimated bonus that participants will be granted if attend the event.

2. `users` table: contains list of users.
   - `user_id` (TEXT, primary key): id for user account.
   - `username` (TEXT)
   - `password` (TEXT): is hashed.
   - `email` (TEXT): user's email.
   - `default_coordinate` (TEXT): the map coordinate of the user's default location.
   - `start_year` (INT): year that the user started as a freshman.
   - `weights` (TEXT, JSON list of 4 FLOAT): weights for 4 parameters in the personal recommendation function, total value of 1.
   - `credits` (TEXT, JSON list of 5 INT): current total credit for each category in the school's rubric.
   - `preferred_topics` (TEXT, JSON list of TEXT): list of ideal topics to be prioritize (used in calculation for `interested` parameter in the personal recommendation function).
   - `preferred_formats` (TEXT, JSON list of TEXT): list of ideal formats to be prioritize (used in calculation for `interested` parameter in the personal recommendation function).

3. `event_subscription` table: contains records of each event subscription from every user.
   - `user_id` (TEXT, primary key, foreign key -> `users.user_id`): id for user account.
   - `event_id` (INT, primary key, foreign key -> `events.event_id`): id for event record.
   - `gain` (FLOAT, [0..1]): estimate credit value.
   - `fit` (FLOAT, [0..1]): schedule fitting and travel distance value.
   - `interested` (FLOAT, [0..1]): topics and formats matching value.
   - `urgency` (FLOAT, [0..1]): deadline to registration or start time value.
   - `role` (TEXT): the role that the user participates as (participant, host, collaborator).
   - `status` (TEXT): (interested, registered, in progress, done).
   - `credit` (INT): the result credit calculated with event's title, description, category, user's role.
   - `other_awards` (TEXT): other awards (certificates, titles,...) or none.
 
4. `fanpages` table: contains list of facebook fanpages.
   - `fanpage_id` (INT, primary key): id for fanpage.
   - `fanpage_name` (TEXT): fanpage name.
   - `number_of_events` (INT): number of events in `events` table.

5. `notifications` table: contains list of notifications 
   - `notification_id` (INT, primary key): id for notification.
   - `user_id` (TEXT, foreign key -> `users.user_id`): id for receiver.
   - `type` (TEXT): notification type (update for events, reminder,...)
   - `content` (TEXT): notification content.
   - `event_id` (INT, foreign key -> `events.event_id`): event id if the notification is about an event being delayed or removed.

6. `rubric` table: contains the school's max credit for each category.
   - `school_id` (INT, primary key): id for school.
   - `max_credit` (TEXT, JSON list of 5 INT): max credit for each category, total of 100.

The system contains the total of 9 modules:
    1. ING - Ingestion: crawl facebook posts through list of fanpages in `fanpages` table in database (3 times a day at 8:00, 14:00, 20:00 +7UTC) and receive inbox from clients as email posts. 
    2. EXT - Extraction: call for LLM models to extract basic informations in the post's raw text and media (include a heuristic keyword extractor fallback), the outcome will be checked as new event, update for an existing one or deduplication. As for new event, the a new record will be added to the `events` table, as the `classified` and `min_credit` field will be left as null.
    3. CLF - Classification: utilizes event record's `title`, `keywords`, `topic`, `format` to decide what category it fits in, update the record's `classified` field.
    4. SCH - Schedule detection: uses event record's `event_start_datetime`, `event_end_datetime` to compare with user's calendar entry, `location` to calculate distance to event's location. These informations will also be calculated into a value as `fit`.
    5. CRD - Credit calculation: calculates result credit using event's title, description, category, user's role to a value as `gain`.
    6. RCM - Recommendation: uses event record's `reg_start_datetime`, `reg_end_datetime` to calculate `urgency` value. Compares the event's `topic`, `format` with the user's `preferred_topic`, `preferred_format` to calculate the `interested` value. Using the user's `weight` values and `gain`, `fit`, `interested`, `urgency`, this module will calculate the priority value for events that appears in user's feed.
    7. PRG - Progress tracking: gives questions to user after registration time, event end time,... to verify the event's status in `event_subscription`.
    8. NTF - Notification: sends notifications to users.
    9. DSH - Dashboard: interact with users, show feeds, summerized dashboard.

The modules work together through the interface of both server (marked as S) and users' local PC (marked as U). Here is the overall workflow.

On server (3 times a day):
1. ING(S) crawls posts and receives inbox then sends to EXT(S) as raw text, EXT update the record in `events` table.
2. CLF(S) queries records in `events` with `category` as null, then run the classify function.
3. CRD(S) queries records in `events` with `min_credit` as null, then run credit_calc(event_id, role) with default role as participant.

On local (when users open the dashboard)
1. SCH(U) matches events' with user's calendar entry, results as (`schedule_free_type`, `distance`, `fit` value).
2. CRD(U) estimate the credit (min(event's `min_credit`, event's `category`'s `max_credit` in `rubric` table - event's `category`'s `credits` in `user` table)), results as `gain` value.
3. RCM(U) calculates `interest`, `urgency`. The event's prioritize score is w = w1 * `gain` + w2 * `fit` + w3 * `interested` + w4 * `urgency` (with w1, w2, w3, w4 as the 4 value in `weights` in `users` table). The score then goes through MMR to reduce bias and diverse the reccommendation for user's feeds.
4. User can browses the feeds and adjust the weights, when user 