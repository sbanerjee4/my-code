import java.util.*;
import java.io.*;
import java.lang.*;

public class track
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("track.in"));
      int n = in.nextInt();
      int time = 0;
      int[] duration = new int[n];
      int[] status = new int[n];
      int[] finish = new int[n];
      int no = n;
      ArrayList<Integer>[] signal = new ArrayList[n];
      for(int i = 0; i < signal.length; i++)
         signal[i] = new ArrayList<Integer>();
      for(int i = 0; i < n; i++)
      {
         duration[i] = in.nextInt();
         int temp = in.nextInt();
         for(int j = 0; j < temp; j++)
            signal[i].add(in.nextInt());         
      }
      
      in.close();
      
      status[0] = 2;
      finish[0] = duration[0];
      no--;
      for(int i = 0; i < signal[0].size(); i++)
      {
         finish[signal[0].get(i) - 1] = duration[0] + duration[signal[0].get(i) - 1];
         status[signal[0].get(i) - 1] = 1;
      }
      while(no > 0)
      {
         int mini = Integer.MAX_VALUE;
         int pos = 0;
         boolean ok = false;
         for(int i = 0; i < finish.length; i++)
         {
            if(status[i] == 1 && finish[i] > time && finish[i] < mini)
            {
               pos = i;
               mini = finish[i];
               ok = true;
            }
         } 
         
         if(ok)
            status[pos] = 2;
         
         no--;
         
         for(int i = 0; i < signal[pos].size(); i++)
            if(status[signal[pos].get(i) - 1] == 0)
            {
               status[signal[pos].get(i) - 1] = 1;
               finish[signal[pos].get(i) - 1] = finish[pos] + duration[signal[pos].get(i) - 1];
            }
      }
      Arrays.sort(finish);
      System.out.println(finish[finish.length - 1]);
   }
}